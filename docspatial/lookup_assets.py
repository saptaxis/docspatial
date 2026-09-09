# lookup_assets

"""Static lookup tables.

CURRENCY_CODES maps an ISO 4217 code to (name, code, symbol). The symbol is
omitted for currencies that have no distinct one, which is why callers guard
with ``len(v) > 2``.
"""

CURRENCY_CODES = {
    "AED": ("UAE Dirham", "AED"),
    "AUD": ("Australian Dollar", "AUD", "$"),
    "BDT": ("Bangladeshi Taka", "BDT", "৳"),
    "BRL": ("Brazilian Real", "BRL", "R$"),
    "CAD": ("Canadian Dollar", "CAD", "$"),
    "CHF": ("Swiss Franc", "CHF"),
    "CNY": ("Chinese Yuan", "CNY", "¥"),
    "DKK": ("Danish Krone", "DKK", "kr"),
    "EUR": ("Euro", "EUR", "€"),
    "GBP": ("Pound Sterling", "GBP", "£"),
    "HKD": ("Hong Kong Dollar", "HKD", "$"),
    "IDR": ("Indonesian Rupiah", "IDR", "Rp"),
    "ILS": ("Israeli New Shekel", "ILS", "₪"),
    "INR": ("Indian Rupee", "INR", "₹"),
    "JPY": ("Japanese Yen", "JPY", "¥"),
    "KRW": ("South Korean Won", "KRW", "₩"),
    "LKR": ("Sri Lankan Rupee", "LKR", "Rs"),
    "MXN": ("Mexican Peso", "MXN", "$"),
    "MYR": ("Malaysian Ringgit", "MYR", "RM"),
    "NGN": ("Nigerian Naira", "NGN", "₦"),
    "NOK": ("Norwegian Krone", "NOK", "kr"),
    "NZD": ("New Zealand Dollar", "NZD", "$"),
    "PHP": ("Philippine Peso", "PHP", "₱"),
    "PKR": ("Pakistani Rupee", "PKR", "Rs"),
    "PLN": ("Polish Zloty", "PLN", "zł"),
    "RUB": ("Russian Ruble", "RUB", "₽"),
    "SAR": ("Saudi Riyal", "SAR"),
    "SEK": ("Swedish Krona", "SEK", "kr"),
    "SGD": ("Singapore Dollar", "SGD", "$"),
    "THB": ("Thai Baht", "THB", "฿"),
    "TRY": ("Turkish Lira", "TRY", "₺"),
    "TWD": ("New Taiwan Dollar", "TWD", "$"),
    "USD": ("United States Dollar", "USD", "$"),
    "VND": ("Vietnamese Dong", "VND", "₫"),
    "ZAR": ("South African Rand", "ZAR", "R"),
}
