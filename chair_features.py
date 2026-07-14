"""Shared, pure feature extractor for #011 empty-chair.

Imported by every pipeline (labels, registry, psc, training, serving) so there is
one definition of every feature and train/serve cannot skew. Two families:

- `registry_features(row, mill_count)`: the shape of a Companies House basic-data
  row at observation time (formation, filing, SIC, name, agent-mill address).
- `psc_features(records)`: the shape of a company's PSC filings (the declared, the
  exempted, the routed-through-a-shell, the silent).

Nothing here calls the network or a feature store. Same input, same output.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ---------------------------------------------------------------- name matching

_SUFFIX = re.compile(r"\b(LIMITED|LTD|LLP|PLC|LP|CO|COMPANY|HOLDINGS|GROUP|THE)\b")


def normalize_name(name: str) -> str:
    """Canonical company-name key for ICIJ<->Companies House matching."""
    s = re.sub(r"[^A-Z0-9 ]", " ", (name or "").upper())
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------- registry features

# holding / management / trust SIC prefixes that concentrate in nominee structures
_HOLDING_SIC = ("64209", "64205", "64303", "70221", "70100", "68209", "68320", "82990")
_DORMANT_SIC = ("99999", "98000")


def _shannon(s: str) -> float:
    s = s.replace(" ", "")
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _year(date_str: str) -> int:
    # Companies House basic dates are dd/mm/yyyy
    m = re.search(r"/(\d{4})$", date_str or "")
    return int(m.group(1)) if m else 0


def primary_sic(row: dict) -> str:
    """Primary SIC code, '' when none. CH writes 'None Supplied' for missing."""
    sic1 = row.get("SICCode.SicText_1", "") or ""
    if sic1.strip().lower().startswith("none supplied"):
        return ""
    return sic1.split(" ", 1)[0].strip()


def registry_features(row: dict, mill_count: int) -> dict:
    """Registry shape for one company. `mill_count` = number of companies sharing
    this company's registered-office address (a formation-mill signal)."""
    name = row.get("CompanyName", "") or ""
    sic_code = primary_sic(row)
    n_sic = sum(1 for k in ("SICCode.SicText_1", "SICCode.SicText_2",
                            "SICCode.SicText_3", "SICCode.SicText_4") if (row.get(k) or "").strip())
    n_prev = sum(1 for i in range(1, 11) if (row.get(f"PreviousName_{i}.CompanyName") or "").strip())
    acct = (row.get("Accounts.AccountCategory", "") or "").upper()

    return {
        "incorporation_year": _year(row.get("IncorporationDate", "")),
        "sic_code": sic_code,
        "sic_section": sic_code[:2] if sic_code else "",
        "is_holding_sic": int(sic_code.startswith(_HOLDING_SIC)),
        "is_dormant_sic": int(sic_code.startswith(_DORMANT_SIC)),
        "n_sic": n_sic,
        "company_category": row.get("CompanyCategory", "") or "",
        "company_status": row.get("CompanyStatus", "") or "",
        "country": (row.get("RegAddress.Country", "") or "").upper(),
        "post_area": re.match(r"[A-Z]+", (row.get("RegAddress.PostCode", "") or "").upper()).group(0)
        if re.match(r"[A-Z]+", (row.get("RegAddress.PostCode", "") or "").upper()) else "",
        # the mill tell: how many companies share this registered office
        "office_company_count": mill_count,
        "is_mill_address": int(mill_count >= 50),
        # filing shape
        "accounts_dormant": int("DORMANT" in acct or "NO ACCOUNTS" in acct),
        "accounts_micro": int("MICRO" in acct),
        "accounts_unaudited": int("UNAUDITED" in acct or "TOTAL EXEMPTION" in acct),
        "n_mortgages": int(row.get("Mortgages.NumMortCharges", "0") or 0),
        # name shape
        "name_len": len(name),
        "name_entropy": round(_shannon(name), 3),
        "name_has_digit": int(bool(re.search(r"\d", name))),
        "n_previous_names": n_prev,
    }


# ------------------------------------------------------------------ psc features

# PSC record kinds in the Companies House snapshot
_STATEMENT_SILENCE = {
    "no-individual-or-entity-with-signficant-control",  # CH's own misspelling, kept verbatim
    "no-individual-or-entity-with-signficant-control-partnership",
    "steps-to-find-psc-not-yet-completed",
    "psc-exists-but-not-identified",
    "psc-details-not-confirmed",
    "psc-contacted-but-no-response",
    "restrictions-notice-issued-to-psc",
}


def psc_features(records: list[dict]) -> dict:
    """Concealment shape from a company's PSC records. `records` are the `data`
    objects from the CH PSC snapshot for one company_number."""
    n = len(records)
    n_individual = n_corporate = n_legal = n_super_secure = n_exempt = 0
    n_statement = n_silence = 0
    foreign_corp = 0
    countries = set()

    for d in records:
        kind = d.get("kind", "")
        if kind.startswith("individual") or "individual" in kind:
            n_individual += 1
            res = (d.get("country_of_residence") or "").upper()
            if res and res not in ("ENGLAND", "SCOTLAND", "WALES", "NORTHERN IRELAND",
                                   "UNITED KINGDOM", "UK", "GREAT BRITAIN"):
                countries.add(res)
        elif kind.startswith("corporate") or "corporate" in kind:
            n_corporate += 1
            ident = d.get("identification", {}) or {}
            reg = (ident.get("country_registered") or ident.get("place_registered") or "").upper()
            if reg and not any(w in reg for w in ("ENGLAND", "WALES", "SCOTLAND", "UNITED KINGDOM", "UK")):
                foreign_corp += 1
                countries.add(reg[:40])
        elif "legal-person" in kind:
            n_legal += 1
        elif "super-secure" in kind:
            n_super_secure += 1
        elif "exemption" in kind or kind == "exemptions":
            n_exempt += 1
        elif "statement" in kind:
            n_statement += 1
            if d.get("statement", "") in _STATEMENT_SILENCE:
                n_silence += 1

    n_real = n_individual + n_corporate + n_legal
    return {
        "psc_n_records": n,
        "psc_n_individual": n_individual,
        "psc_n_corporate": n_corporate,
        "psc_n_legal_person": n_legal,
        # the core concealment tells
        "psc_has_no_individual": int(n_individual == 0),           # no natural owner declared
        "psc_corporate_only": int(n_corporate > 0 and n_individual == 0),
        "psc_corporate_ratio": round(n_corporate / n_real, 3) if n_real else 0.0,
        "psc_foreign_corporate": foreign_corp,
        "psc_n_statement": n_statement,
        "psc_silence": int(n_silence > 0),                          # declared "no PSC" / not-identified
        "psc_super_secure": int(n_super_secure > 0),                # protected identity
        "psc_exempt": int(n_exempt > 0),                            # claimed exemption
        "psc_absent": int(n_real == 0),                             # nobody real on file at all
        "psc_n_foreign_countries": len(countries),
    }


# the model's feature contract: the exact columns and order train and serve share
CAT_FEATURES = ["company_category", "company_status", "country", "post_area", "sic_section"]
NUM_FEATURES = [
    "incorporation_year", "is_holding_sic", "is_dormant_sic", "n_sic",
    "office_company_count", "is_mill_address", "accounts_dormant", "accounts_micro",
    "accounts_unaudited", "n_mortgages", "name_len", "name_entropy", "name_has_digit",
    "n_previous_names",
    "psc_n_records", "psc_n_individual", "psc_n_corporate", "psc_n_legal_person",
    "psc_has_no_individual", "psc_corporate_only", "psc_corporate_ratio",
    "psc_foreign_corporate", "psc_n_statement", "psc_silence", "psc_super_secure",
    "psc_exempt", "psc_absent", "psc_n_foreign_countries",
]
MODEL_FEATURES = CAT_FEATURES + NUM_FEATURES

# the concealment tells the app surfaces as fired evidence, with plain labels
CONCEALMENT_FLAGS = {
    "psc_absent": "No real person or entity with significant control on file",
    "psc_silence": "Declared no-PSC / not-identified / steps-not-completed",
    "psc_corporate_only": "Ownership routed only through corporate entities",
    "psc_foreign_corporate": "Corporate owner registered outside the UK",
    "psc_super_secure": "Owner identity is protected (super-secure)",
    "psc_exempt": "Company claims a PSC exemption",
    "is_mill_address": "Registered at a formation-mill address",
    "is_holding_sic": "Holding / management / trust activity code",
    "accounts_dormant": "Files dormant / no-trading accounts",
}

# v4 model contract: interaction features + frozen target encodings, derived from
# the disclosure columns only. ONE code path for training and every scoring pass;
# the te_maps ship inside the model artifact (te_maps.json).
DERIVED_NUM = ["n_tells", "mill_x_no_individual", "mill_x_silence",
               "dormant_x_corp_only", "holding_x_foreign", "foreign_corp_ratio"]
# SIC encodings dropped after the v5/v6 bias audits: any sector TE lets the model
# rank on sector directly and the universe top 1% collapses to real estate
# (99.3% with sic_code, still 88.3% with sic_section only). docs/bias-audit.md.
TE_COLS = ["post_area"]
_TELLS = ["psc_absent", "psc_silence", "psc_corporate_only", "psc_super_secure",
          "psc_exempt", "is_mill_address", "is_holding_sic", "accounts_dormant"]


def derive_features(df, te_maps=None):
    """Add DERIVED_NUM interactions and, when te_maps is given, the <col>_te
    encodings. Mutates and returns df."""
    df["n_tells"] = df[_TELLS].sum(axis=1)
    df["mill_x_no_individual"] = df["is_mill_address"] * df["psc_has_no_individual"]
    df["mill_x_silence"] = df["is_mill_address"] * df["psc_silence"]
    df["dormant_x_corp_only"] = df["accounts_dormant"] * df["psc_corporate_only"]
    df["holding_x_foreign"] = df["is_holding_sic"] * (df["psc_foreign_corporate"] > 0).astype(int)
    df["foreign_corp_ratio"] = df["psc_foreign_corporate"] / df["psc_n_corporate"].clip(lower=1)
    if te_maps:
        prior = te_maps["__prior__"]
        # the artifact's te_maps.json says which columns were encoded, so older
        # models (e.g. v5 with sic_code) keep scoring after TE_COLS changes
        for c in te_maps:
            if c != "__prior__":
                df[c + "_te"] = df[c].astype(str).map(te_maps[c]).fillna(prior)
    return df


# feature-group descriptions, the published contract
REGISTRY_DOC = {
    "incorporation_year": "Year the company was incorporated",
    "sic_code": "Primary SIC code",
    "sic_section": "First two digits of the primary SIC code",
    "is_holding_sic": "Primary SIC is a holding/management/trust code",
    "is_dormant_sic": "Primary SIC marks a dormant/non-trading company",
    "n_sic": "Number of SIC codes declared (1-4)",
    "company_category": "Companies House company category",
    "company_status": "Active, Dissolved, Liquidation, ...",
    "country": "Registered-office country",
    "post_area": "Alphabetic prefix of the registered postcode",
    "office_company_count": "Companies sharing this registered-office address (mill signal)",
    "is_mill_address": "Registered office shared by >=50 companies",
    "office_group": "Stable id of the registered-office address, for grouped train/test splits",
    "accounts_dormant": "Latest accounts category is dormant / none",
    "accounts_micro": "Latest accounts category is micro-entity",
    "accounts_unaudited": "Latest accounts are unaudited / total-exemption",
    "n_mortgages": "Number of mortgage charges ever registered",
    "name_len": "Company-name character length",
    "name_entropy": "Shannon entropy of the company name",
    "name_has_digit": "Company name contains a digit",
    "n_previous_names": "Count of previous names on file",
}

PSC_DOC = {
    "psc_n_records": "Total PSC records on file",
    "psc_n_individual": "Individual PSCs",
    "psc_n_corporate": "Corporate-entity PSCs",
    "psc_n_legal_person": "Legal-person PSCs",
    "psc_has_no_individual": "No natural-person PSC declared",
    "psc_corporate_only": "Ownership routed only through corporate entities",
    "psc_corporate_ratio": "Corporate share of real PSCs",
    "psc_foreign_corporate": "Corporate PSCs registered outside the UK",
    "psc_n_statement": "Count of PSC statement records",
    "psc_silence": "Declared no-PSC / not-identified / steps-not-completed",
    "psc_super_secure": "Protected (super-secure) PSC on file",
    "psc_exempt": "Company claims a PSC exemption",
    "psc_absent": "No real PSC (individual/corporate/legal) on file at all",
    "psc_n_foreign_countries": "Distinct foreign residencies/registrations among PSCs",
}
