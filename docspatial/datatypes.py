# datatypes.py
import re
from enum import Enum

import datefinder
import dateparser
import dateparser.search
import dateutil
import usaddress
from assets import load_assets
from nameparser import HumanName

from . import lookup_assets, text_utils


class DataTypes(Enum):
    DATE = "date"
    NUMBER_QUANT = "number__quantitative"
    NUMBER = "number"
    ID = "id"
    ADDRESS = "address"
    FREEFORM = "freeform"
    EMPTY = "empty"

    # def __eq__(self, other):
    #     # comparing empty with anything else will return True
    #     # if self is DataTypes.EMPTY or other is DataTypes.EMPTY:
    #     #     return True
    #     return super().__eq__(other)

    def matches(self, other, match_empty=False):
        if match_empty and (self is DataTypes.EMPTY or other is DataTypes.EMPTY):
            return True
        return self is other

    @classmethod
    def compare_datatype_list(cls, list1, list2, match_empty=False):
        if (not list1) or (not list2):
            return 0
        if len(list1) != len(list2):
            return 0
        matches = 0
        for l1, l2 in zip(list1, list2):
            this_match = l1.matches(l2, match_empty=match_empty)
            if this_match:
                matches += 1
        total = len(list1)
        match_fraction = matches / total
        return match_fraction


class DataTypeFinder:
    def __init__(self):
        self.address_dt_obj = Address()

    def find_datatype(self, text):
        words_text = text_utils.split_text_into_words(text, split_punctuation=True)

        is_date = Date(text).is_date()
        is_quant_number = text_utils.is_readable_number(text, quantifiable_number=True)
        is_id = text_utils.is_id_code(text, only_upper=True)
        is_generic_number = text_utils.is_readable_number(text)
        is_address = self.address_dt_obj.is_address(words_text)

        if not text:
            return DataTypes.EMPTY
        if is_quant_number:
            return DataTypes.NUMBER_QUANT
        if is_id:
            return DataTypes.ID
        if is_generic_number:
            return DataTypes.NUMBER
        if is_date:
            return DataTypes.DATE
        if is_address:
            return DataTypes.ADDRESS
        return DataTypes.FREEFORM


def create_datatype_vector(text, address_dt_obj=None):
    if text is None:
        text = ""

    if address_dt_obj is None:
        address_dt_obj = Address()

    words_text = text_utils.split_text_into_words(text, split_punctuation=True)

    # Dates
    contains_date = Date(text).contains_date(parts=["year"])
    found_dates = Date(text).search_dates(text, parts=["year"])
    found_dates = "" if found_dates is None else found_dates

    # ID
    contains_id = False
    for w in words_text:
        if AlphaNumeric(w).is_alphanumeric():
            contains_id = True

    # Numeric
    # remove detected dates and try numeric
    all_found_dates = " ".join([ii[0] for ii in found_dates])
    words_text_filtered = []
    for w in words_text:
        if w not in all_found_dates:
            words_text_filtered.append(w)
    contains_numeric = False
    for w in words_text_filtered:
        if Numeric(w).is_numeric():
            contains_numeric = True

    # Address
    contains_address = address_dt_obj.is_address(words_text)

    # short phrase
    is_short_phrase = len(words_text) < 10

    # long phrase
    is_long_phrase = len(words_text) >= 10

    dt_vector = [
        1 if c else 0
        for c in [
            contains_date,
            contains_id,
            contains_numeric,
            contains_address,
            is_short_phrase,
            is_long_phrase,
        ]
    ]
    return dt_vector


class AlphaNumeric:
    def __init__(self, text, num_chars=None):
        if text is None:
            text = ""
        self.text = text.strip()
        self.num_chars = num_chars

    def is_alphanumeric(self):
        is_alphanum = text_utils.is_id_code(self.text)
        len_check = True
        if self.num_chars:
            len_check = len(self.text) == self.num_chars
        return is_alphanum and len_check


class Numeric:
    number_style_map = [
        "Integer",
        "Decimal Rounded to Two Digits",
        "Decimal Rounded to Three Digits",
        "International Comma Separated",
        "Indian Comma Separated",
        "International Comma Separated, Decimal Rounded to Two Digits",
        "International Comma Separated, Decimal Rounded to Three Digits",
        "Indian Comma Separated, Decimal Rounded to Two Digits",
        "Indian Comma Separated, Decimal Rounded to Three Digits",
    ]

    def __init__(self, text, style=None, quantifiable_number=False):
        if text is None:
            text = ""
        self.text = text.strip()
        self.number = None
        self.formatted_number = None
        self.quantifiable_number = quantifiable_number
        if self.is_numeric(quantifiable_number=self.quantifiable_number):
            self.number = float(self.text.replace(",", ""))
            if style is not None:
                self.formatted_number = self.format_number(style)

    def is_numeric(self, quantifiable_number=False):
        return text_utils.is_readable_number(
            self.text, quantifiable_number=quantifiable_number
        )

    @classmethod
    def convert_to_indian_format(cls, number):
        num_str = str(number)

        # store decimal component
        decimal = False
        if "." in num_str:
            decimal = True
            main, fraction = num_str.split(".")
        else:
            main = num_str

        # get the last 3 digits to store as a chunk
        last_chunk = main[-3:]

        # get the remaining chunks
        first_chunks = main[:-3]

        if len(first_chunks):
            # invert the chunks
            first_chunks_inv = first_chunks[::-1]
            # regex to add commas with 2 digits only if look ahead number exists
            first_chunks_inv = re.sub(r"(\d{2})(?=\d)", r"\1,", first_chunks_inv)
            # invert it back
            first_chunks = first_chunks_inv[::-1]
            # add it to the 3 digit chunk
            num_str_comma = first_chunks + "," + last_chunk
        else:
            num_str_comma = last_chunk
        # add back the decimal
        if decimal:
            num_str_comma += "." + fraction
        return num_str_comma

    def format_number(self, style):
        if style not in self.number_style_map:
            return self.text
        if self.number is None:
            return self.text

        try:
            if style == "Integer":
                return str(int(self.number))
            elif style == "International Comma Separated":
                return "{:,}".format(self.number)
            elif style == "Indian Comma Separated":
                return self.convert_to_indian_format(self.number)
            elif style == "Decimal Rounded to Two Digits":
                return "{:.2f}".format(self.number)
            elif style == "Decimal Rounded to Three Digits":
                return "{:.3f}".format(self.number)
            elif (
                style == "International Comma Separated, Decimal Rounded to Two Digits"
            ):
                main, decimals = str(self.number).split(".")
                main_comma = "{:,}".format(int(main))
                decimal_fake = float(f"0.{decimals}")
                decimal_fake_rounded = "{:.2f}".format(decimal_fake)
                decimals_rounded = decimal_fake_rounded.split(".")[1]
                return f"{main_comma}.{decimals_rounded}"
            elif (
                style
                == "International Comma Separated, Decimal Rounded to Three Digits"
            ):
                main, decimals = str(self.number).split(".")
                main_comma = "{:,}".format(int(main))
                decimal_fake = float(f"0.{decimals}")
                decimal_fake_rounded = "{:.3f}".format(decimal_fake)
                decimals_rounded = decimal_fake_rounded.split(".")[1]
                return f"{main_comma}.{decimals_rounded}"
            elif style == "Indian Comma Separated, Decimal Rounded to Two Digits":
                main, decimals = str(self.number).split(".")
                main_comma = self.convert_to_indian_format(int(main))
                decimal_fake = float(f"0.{decimals}")
                decimal_fake_rounded = "{:.2f}".format(decimal_fake)
                decimals_rounded = decimal_fake_rounded.split(".")[1]
                return f"{main_comma}.{decimals_rounded}"
            elif style == "Indian Comma Separated, Decimal Rounded to Three Digits":
                main, decimals = str(self.number).split(".")
                main_comma = self.convert_to_indian_format(int(main))
                decimal_fake = float(f"0.{decimals}")
                decimal_fake_rounded = "{:.3f}".format(decimal_fake)
                decimals_rounded = decimal_fake_rounded.split(".")[1]
                return f"{main_comma}.{decimals_rounded}"
        except Exception as e:
            return self.text


class USAddress:
    us_state_codes = {
        "AL": "ALABAMA",
        "AK": "ALASKA",
        "AZ": "ARIZONA",
        "AR": "ARKANSAS",
        "CA": "CALIFORNIA",
        "CO": "COLORADO",
        "CT": "CONNECTICUT",
        "DE": "DELAWARE",
        "FL": "FLORIDA",
        "GA": "GEORGIA",
        "HI": "HAWAII",
        "ID": "IDAHO",
        "IL": "ILLINOIS",
        "IN": "INDIANA",
        "IA": "IOWA",
        "KS": "KANSAS",
        "KY": "KENTUCKY",
        "LA": "LOUISIANA",
        "ME": "MAINE",
        "MD": "MARYLAND",
        "MA": "MASSACHUSETTS",
        "MI": "MICHIGAN",
        "MN": "MINNESOTA",
        "MS": "MISSISSIPPI",
        "MO": "MISSOURI",
        "MT": "MONTANA",
        "NE": "NEBRASKA",
        "NV": "NEVADA",
        "NH": "NEW HAMPSHIRE",
        "NJ": "NEW JERSEY",
        "NM": "NEW MEXICO",
        "NY": "NEW YORK",
        "NC": "NORTH CAROLINA",
        "ND": "NORTH DAKOTA",
        "OH": "OHIO",
        "OK": "OKLAHOMA",
        "OR": "OREGON",
        "PA": "PENNSYLVANIA",
        "RI": "RHODE ISLAND",
        "SC": "SOUTH CAROLINA",
        "SD": "SOUTH DAKOTA",
        "TN": "TENNESSEE",
        "TX": "TEXAS",
        "UT": "UTAH",
        "VT": "VERMONT",
        "VA": "VIRGINIA",
        "WA": "WASHINGTON",
        "WV": "WEST VIRGINIA",
        "WI": "WISCONSIN",
        "WY": "WYOMING",
    }

    def __init__(self, text, state_expanded=True):
        if text is None:
            text = ""
        self.text = text.strip()
        self.state_expanded = state_expanded

        self._expand_address()

        self.full_address = self._is_full_us_address()

    def _is_full_us_address(self):
        found_list = []
        for attr in [
            "street",
            "city",
            "state",
            "zip",
        ]:
            found_list.append(getattr(self, attr))
        return all(found_list)

    def _expand_address(self):
        # using package usaddress
        for attr in [
            "house_number",
            "direction",
            "street",
            "unit",
            "city",
            "state",
            "zip",
            "zip_plus",
            "first_line",
            "address_type",
        ]:
            setattr(self, attr, "")
        # each value itself can be a list if usaddress.parse is used
        # if usaddress.tag is used, it will be merged
        try:
            parsed_address, address_type = usaddress.tag(self.text)
        except Exception as e:
            # this often means not a very clean address string
            return

        usaddress_mapper = {
            "house_number": "AddressNumber",
            "direction": "StreetNamePreDirectional",
            "street": ["StreetName", "StreetNamePostType"],
            "unit": ["OccupancyType", "OccupancyIdentifier"],
            "city": "PlaceName",
            "state": "StateName",
            "zip": "ZipCode",
        }

        address_values = {}
        for address_part, usaddress_attr in usaddress_mapper.items():
            if isinstance(usaddress_attr, str):
                this_val = parsed_address.get(usaddress_attr, "")
            elif isinstance(usaddress_attr, list):
                this_val = " ".join(
                    [parsed_address.get(usattr, "") for usattr in usaddress_attr]
                )
            this_val = this_val.strip()
            address_values[address_part] = this_val

        # address first line
        attrs_to_join = ["house_number", "direction", "street"]
        address_values["first_line"] = " ".join(
            [address_values.get(aa, "") for aa in attrs_to_join]
        )

        # state expansion
        if self.state_expanded:
            state_pred = address_values.get("state", "")
            state_expanded = self.us_state_codes.get(state_pred.upper())
            if state_expanded:
                address_values["state"] = state_expanded
        else:
            state_codes_inv = {v: k for k, v in self.us_state_codes.items()}
            state_pred = address_values.get("state", "")
            state_contracted = state_codes_inv.get(state_pred.upper())
            if state_contracted:
                address_values["state"] = state_contracted

        # zip split
        address_values["zip_plus"] = ""
        zip_pred = address_values.get("zip", "")
        if (len(zip_pred) > 5) and ("-" in zip_pred):
            zip_splits = zip_pred.split("-")

            if len(zip_splits[0]) == 5:
                address_values["zip"] = zip_splits[0].strip()
            if len(zip_splits[1]) == 4:
                address_values["zip_plus"] = zip_splits[1].strip()

        # Direction Normalization
        direction_pred = address_values.get("direction", "")
        if direction_pred:
            direction_pred = direction_pred.upper()
            if direction_pred in ["N", "NORTH"]:
                address_values["direction"] = "NORTH-N"
            elif direction_pred in ["S", "SOUTH"]:
                address_values["direction"] = "SOUTH-S"
            elif direction_pred in ["E", "EAST"]:
                address_values["direction"] = "EAST-E"
            elif direction_pred in ["W", "WEST"]:
                address_values["direction"] = "WEST-W"

        # setattrs
        for k, v in address_values.items():
            setattr(self, k, v)
        self.address_type = address_type

    def _to_dict(self, prepend="address"):
        output_dict = {}
        for attr in [
            "house_number",
            "direction",
            "street",
            "unit",
            "city",
            "state",
            "zip",
            "zip_plus",
            "first_line",
        ]:
            output_dict[f"{prepend}__{attr}"] = getattr(self, attr)
        return output_dict


class Name:
    name_style_map = [
        "First Name Only",
        "Last Name Only",
        "Full Name",
        "Last Name, First Name",
        "Initials. Last Name",
    ]

    def __init__(self, text, style=None, validate=False):
        if text is None:
            text = ""
        self.text = text.strip()
        self.human_name = HumanName(self.text, initials_format="{first} {middle}")

        self.formatted_name = text
        if style is not None:
            self.formatted_name = self.format_name(style)

        if validate and not self.is_name():
            self.human_name = None
            self.formatted_name = None

        self._expand_name()

    def _expand_name(self):
        self.full_name = None
        self.title = None
        self.first_name = None
        self.middle_name = None
        self.last_name = None
        self.suffix = None

        if self.human_name is None:
            return

        self.full_name = self.human_name.full_name
        self.title = self.human_name.title
        self.first_name = self.human_name.first
        self.middle_name = self.human_name.middle
        self.last_name = self.human_name.last
        self.suffix = self.human_name.suffix

    def _to_dict(self, middle_initial_in_first=False):
        output_dict = {
            "name": self.text,
            "name__full": self.human_name.full_name,
            "name__title": self.title,
            "name__first": self.first_name,
            "name__middle": self.middle_name,
            "name__last": self.last_name,
            "name__suffix": self.suffix,
        }

        if middle_initial_in_first and len(self.middle_name):
            output_dict["name__first"] = (
                output_dict["name__first"]
                + " "
                + " ".join([ii[0].upper() for ii in self.middle_name.split(" ")])
            )
        return output_dict

    def is_name(self):
        # need something more sophisticated than this
        # could potentially use NER
        is_only_name_chars = re.match(r"^[A-Za-z\s,.\-]+$", self.text) is not None
        words_count_check = 0 < len(self.text.split()) <= 6
        return is_only_name_chars and words_count_check

    def format_name(self, style):
        if style not in self.name_style_map:
            return self.text

        if self.human_name is None:
            return self.text

        try:
            if style == "Full Name":
                return self.human_name.full_name
            elif style == "First Name Only":
                return self.human_name.first
            elif style == "Last Name Only":
                return self.human_name.last
            elif style == "Last Name, First Name":
                return f"{self.human_name.last}, {self.human_name.first}"
            elif style == "Initials. Last Name":
                return f"{self.human_name.initials()} {self.human_name.last}"
        except Exception as e:
            return self.human_name.full_name


class Date:
    date_style_map = {
        # day, month, year
        # day first
        "dd-mm-yyyy": "%d-%m-%Y",
        "dd/mm/yyyy": "%d/%m/%Y",
        "dd mmm yyyy": "%d %b %Y",
        "dd month yyyy": "%d %B %Y",
        # month first
        "mm-dd-yyyy": "%m-%d-%Y",
        "mm/dd/yyyy": "%m/%d/%Y",
        "mmm dd yyyy": "%b %d %Y",
        "mmm dd, yyyy": "%b %d, %Y",
        "month dd, yyyy": "%B %d, %Y",
        # year first
        "yyyy-mm-dd": "%Y-%m-%d",
        "yyyy/mm/dd": "%Y/%m/%d",
        "yyyy mmm dd": "%Y %b %d",
        "yyyy month dd": "%Y %B %d",
        # only month and year
        # month first
        "mm/yyyy": "%m/%Y",
        "mm-yyyy": "%m-%Y",
        "mmm yyyy": "%b %Y",
        "month yyyy": "%B %Y",
        # year first
        "yyyy-mm": "%Y-%m",
        "yyyy/mm": "%Y/%m",
        "yyyy mmm": "%Y %b",
        "yyyy month": "%Y %B",
        # only year
        "yyyy": "%Y",
    }

    def __init__(self, text, style=None):
        if text is None:
            text = ""
        self.text = text.strip()
        self.dt = None
        self.formatted_date = None

        if self.is_date():
            self.dt = dateparser.parse(self.text)
            if style:
                self.formatted_date = self.format_date(self.dt, style)

    def __str__(self):
        return self.text

    def is_date(self):
        """Check if the text is only a date

        Use dateutil.parser to check if the text is a date
        This will fail on anything except a string is pretty much only a date
        """
        return self.get_date_dt() is not None

    def contains_date(self, parts=[], strict_parsing=True):
        """Check if the text contains a date

        Use dateparser.parse to check if the text contains a date
        """
        found_dates = self.search_dates(
            self.text, parts=parts, strict_parsing=strict_parsing
        )
        return found_dates is not None

    def get_date_dt(self):
        text = text_utils.remove_spaces_after_chars(self.text, chars=["-", "/", ","])
        try:
            dt = dateutil.parser.parse(text)
        except Exception as e:
            dt = None
        return dt

    def find_all_date_dt(self, parts=[], method="datefinder", strict_parsing=True):
        if method not in ["datefinder", "dateparser"]:
            raise ValueError("method should be either 'datefinder' or 'dateparser'")
        if method == "datefinder":
            return self.search_dates__datefinder(
                self.text, strict_parsing=strict_parsing
            )
        else:
            return self.search_dates__dateparser(
                self.text, parts=parts, strict_parsing=strict_parsing
            )

    @classmethod
    def format_date(cls, dt, style):
        if style not in cls.date_style_map:
            return None
        try:
            formatted_date = dt.strftime(cls.date_style_map[style])
        except Exception as e:
            formatted_date = dt.strftime("%d %B %Y")
        return formatted_date

    @classmethod
    def search_dates__dateparser(cls, text, parts=[], strict_parsing=True):
        """Search for dates in the text

        Use dateparser.search to find dates in the text

        parts: list of strings (day, month, year)
        """
        # dataparser works better if spaces are removed after typical date separators
        text = text_utils.remove_spaces_after_chars(text, chars=["-", "/", ","])

        settings = {"STRICT_PARSING": strict_parsing, "PARSERS": ["absolute-time"]}
        if parts:
            settings["REQUIRE_PARTS"] = parts

        found_dates = dateparser.search.search_dates(text, settings=settings)
        return found_dates

    def search_dates__datefinder(cls, text, strict_parsing=True):
        """Search for dates in the text

        Use datefinder to find dates in the text
        """
        found_dates = datefinder.find_dates(text, strict=strict_parsing, source=True)

        found_dates_ = []
        for match in found_dates:
            found_dates_.append((match[1], match[0]))

        return found_dates_


class Address:
    def __init__(self):
        self.countries_asset = load_assets.load("geoname_countries")
        self.countries_list = self.countries_asset["Country"].str.lower().tolist()

    def is_address(self, text):
        return is_address(text, countries_asset=self.countries_asset)


def is_org(string):
    # Pattern for organization names
    org_pattern = r"\b(inc|llc|ltd|limited|co|corp|corporation|s\.a\.e|mansfield|pllc|llp|pc|llpattorneys|dunkiel|ltda|office depot)\b"

    if re.search(org_pattern, string, re.IGNORECASE):
        return True
    return False


def is_currency(string):
    # Extracting abbreviations and symbols, ensuring uniqueness
    abbreviations = set(v[1] for v in lookup_assets.CURRENCY_CODES.values())
    symbols = set(v[2] for v in lookup_assets.CURRENCY_CODES.values() if len(v) > 2)

    # Escaping special characters in symbols
    escaped_symbols = [re.escape(symbol) for symbol in symbols]

    # Creating regex patterns
    abbr_pattern = r"\b(?:{})\b".format("|".join(abbreviations))
    symbol_pattern = r"(?:{})".format(
        "|".join(escaped_symbols)
    )  # Removed word boundaries for symbols

    # commented other condition because of currency misclassification
    if re.search(abbr_pattern, string) or re.search(symbol_pattern, string):
        return True
    return False


def has_postal_code(string):
    patterns = {
        "Argentina": r"[A-Z]\d{4}[A-Z]{3}",
        "Brazil": r"\d{5}-\d{3}",
        "Chile": r"\b(\d{7}|\d{3}-\d{4})\b",
        "Colombia": r"\b\d{6}\b",
        "India": r"\b\d{6}\b",
        "Mexico": r"\b\d{5}\b",
        "Peru": r"\b\d{5}\b",
        "Venezuela": r"\b\d{4}[A-Z]?\b",
        "USA": r"\b\d{5}(-\d{4})?\b",
        "Canada": r"\b[A-Z]\d[A-Z] \d[A-Z]\d\b",
    }

    possible_countries = []
    for country, pattern in patterns.items():
        if re.search(pattern, string):
            possible_countries.append(country)
    return possible_countries


def is_email(s):
    regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return re.match(regex, s) is not None


def is_url(text):
    # Regular expression pattern to detect URLs starting with www

    pattern = r"\bwww\.[a-zA-Z0-9]+\.[a-zA-Z]+\b"

    # Find all matches of URLs in the text
    urls = re.findall(pattern, text)
    is_url = True if len(urls) > 0 else False

    return is_url


def is_phone_number(text):
    # List of regex patterns for different phone number conventions
    patterns = [
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US: (123) 456-7890
        r"\+\d{1,2}\s?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}",  # International: +1 234-567-8900
        r"\(?\d{3,4}\)?[\s-]?\d{2,3}[\s-]?\d{2,4}[\s-]?\d{2,4}",  # Other formats: (868) 228 01 02
        # Add additional patterns here as needed
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def contains_address_street(text):
    street_pattern = r"\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|circle|cir)\b"
    street_found = re.search(street_pattern, text, re.IGNORECASE)
    return street_found


def contains_address_country(text, countries_asset=None):
    if isinstance(text, list):
        # list of words
        words = text
        text = " ".join(words)
    elif isinstance(text, str):
        words = text.split()
    else:
        raise ValueError("text should be str or list of words")

    if countries_asset is None:
        countries_asset = load_assets.load("geoname_countries")
    countries_list = countries_asset["Country"].str.lower().tolist()
    contains_country = any([w.lower() in countries_list for w in words])
    return contains_country


def is_address(text, countries_asset=None):
    # check if country exists
    # check if is street
    # postal code - NOT Done
    if isinstance(text, list):
        # list of words
        words = text
        text = " ".join(words)
    elif isinstance(text, str):
        words = text.split()
    else:
        raise ValueError("text should be str or list of words")
    has_country = contains_address_country(text, countries_asset=countries_asset)
    has_street = contains_address_street(text)
    return has_country and has_street


def is_address_old(string, debug=False):
    # Pattern for street type (street, avenue, road, etc.)

    # Pattern for state/province abbreviation (assuming two uppercase letters)
    state_pattern = r"\b[A-Z]{2}\b"

    # Refined pattern for building number
    building_number_pattern = r"\b\d{1,4}\b\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|circle|cir)\b"

    # Pattern to exclude common date formats and code-like strings
    date_pattern = r"\b\d{1,4}[-\/]\d{1,2}[-\/]\d{1,4}\b"
    code_pattern = r"\b([A-Za-z]+-)+[A-Za-z0-9]+\b"

    street_found = contains_address_street(string)
    state_found = re.search(state_pattern, string)
    postal_code_found = has_postal_code(string)
    building_number_found = re.search(building_number_pattern, string, re.IGNORECASE)
    date_found = re.search(date_pattern, string)
    code_found = re.search(code_pattern, string)
    city_found = re.search(r"^[a-zA-Z\s]+$", string)

    if debug:
        print(
            f"Street: {street_found}, State: {state_found}, Postal Code: {postal_code_found}, Building Number: {building_number_found}, Date: {date_found}, Code: {code_found}, City: {city_found}"
        )

    if (
        (street_found or state_found or postal_code_found or building_number_found)
        and not date_found
        and not code_found
    ):
        return True
    elif city_found:  # Simple check for city names
        return True
    return False
