"""Hand-written personal-finance questions. Ground truth is computed by the reference
code next to each question (never typed in by hand), so the labels are exact."""

import math


def pmt(principal, annual_rate, years, periods_per_year=12):
    r = annual_rate / periods_per_year
    n = years * periods_per_year
    return principal * r / (1 - (1 + r) ** -n)


def fv_monthly(deposit, annual_rate, years):
    r = annual_rate / 12
    n = years * 12
    return deposit * ((1 + r) ** n - 1) / r


def payoff_months(principal, annual_rate, payment):
    r = annual_rate / 12
    return math.ceil(-math.log(1 - r * principal / payment) / math.log(1 + r))


def total_interest(principal, annual_rate, years):
    return pmt(principal, annual_rate, years) * years * 12 - principal


FINANCE_QUESTIONS = [
    ("What is the monthly payment on a $400,000 mortgage at 6.5% annual interest over 30 years?",
     pmt(400_000, 0.065, 30)),
    ("I'm borrowing $250,000 for a home at 5.75% APR for 15 years. What's my monthly payment?",
     pmt(250_000, 0.0575, 15)),
    ("A $32,000 car loan at 7.2% APR for 5 years. What is the monthly payment?",
     pmt(32_000, 0.072, 5)),
    ("How much total interest will I pay over the life of a $400,000, 30-year mortgage at 6.5%?",
     total_interest(400_000, 0.065, 30)),
    ("If I invest $10,000 at 7% interest compounded annually, how much will I have after 20 years?",
     10_000 * 1.07 ** 20),
    ("I deposit $5,000 in a savings account paying 4.5% annual interest compounded monthly. What's the balance after 3 years?",
     5_000 * (1 + 0.045 / 12) ** 36),
    ("If I save $500 every month at 6% annual interest compounded monthly, how much will I have after 25 years?",
     fv_monthly(500, 0.06, 25)),
    ("I have $18,000 of credit card debt at 22.9% APR and pay $600 a month. How many months until it is paid off?",
     payoff_months(18_000, 0.229, 600)),
    ("A $15,000 student loan at 5.5% APR with $200 monthly payments: how many months to pay it off?",
     payoff_months(15_000, 0.055, 200)),
    ("How many years does it take money to double at 8% annual interest compounded annually? Give the exact value, not the rule of 72.",
     math.log(2) / math.log(1.08)),
    ("Inflation averages 3% per year. What will $100,000 today be worth in purchasing power after 15 years?",
     100_000 / 1.03 ** 15),
    ("I want $1,000,000 in 30 years. Earning 7% compounded monthly, how much must I save each month?",
     1_000_000 * (0.07 / 12) / ((1 + 0.07 / 12) ** 360 - 1)),
    ("A $300,000 mortgage at 6% for 30 years: how much principal is still owed after the first 5 years of payments?",
     300_000 * (1 + 0.005) ** 60 - pmt(300_000, 0.06, 30) * ((1 + 0.005) ** 60 - 1) / 0.005),
    ("Comparing a 30-year and a 15-year $350,000 mortgage, both at 6%: how much total interest does the 15-year loan save?",
     total_interest(350_000, 0.06, 30) - total_interest(350_000, 0.06, 15)),
    ("A stock bought at $84.50 is sold for $112.30 three years later. What is the compound annual growth rate, in percent?",
     ((112.30 / 84.50) ** (1 / 3) - 1) * 100),
]
