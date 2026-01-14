"""
OFAC Sanctions Screening Module.

Implements screening against OFAC SDN (Specially Designated Nationals) list
and other watchlists for banking compliance.

Note: This is a mock implementation for demonstration purposes.
In production, integrate with official OFAC data feeds or commercial
screening services (e.g., Dow Jones, LexisNexis, World-Check).
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class ScreeningResult:
    """Result of a watchlist screening."""
    is_match: bool
    match_type: str  # "EXACT", "FUZZY", "PARTIAL", "NO_MATCH"
    matched_entry: Optional[str]
    list_name: str
    risk_score: float  # 0.0 to 1.0
    reason_code: str
    description: str


class OFACScreener:
    """
    OFAC SDN List Screener.
    
    Screens names and countries against sanctions lists:
    - SDN (Specially Designated Nationals)
    - Sanctioned Countries
    - High-Risk Jurisdictions
    """
    
    # Mock OFAC SDN List (In production, use official OFAC data)
    # Format: (name, aliases, type, program)
    SDN_LIST = [
        ("AHMADI, Mohammad", ["M. AHMADI", "MOHAMMAD AHMADI"], "INDIVIDUAL", "IRAN"),
        ("BANCO DELTA ASIA", ["BDA", "DELTA ASIA BANK"], "ENTITY", "NORTH-KOREA"),
        ("DELTA OIL TRADING", ["DELTA OIL", "DOT"], "ENTITY", "SYRIA"),
        ("PETROV, Alexei", ["A. PETROV", "ALEXEI PETROV"], "INDIVIDUAL", "RUSSIA-UKRAINE"),
        ("CHEN, Wei", ["CHEN WEI", "W. CHEN"], "INDIVIDUAL", "CHINA-MILITARY"),
        ("GOLDEN TRADE LLC", ["GOLDEN TRADING"], "ENTITY", "IRAN"),
        ("HASSAN, Ahmed", ["A. HASSAN", "AHMED HASSAN"], "INDIVIDUAL", "SUDAN"),
        ("NORTH STAR SHIPPING", ["NS SHIPPING"], "ENTITY", "NORTH-KOREA"),
        ("KUMAR, Rajesh", ["R. KUMAR"], "INDIVIDUAL", "INDIA-BLACKLIST"),
        ("SILVA, Carlos", ["C. SILVA", "CARLOS SILVA"], "INDIVIDUAL", "VENEZUELA")
    ]
    
    # Sanctioned Countries (Full embargo)
    SANCTIONED_COUNTRIES = {
        "NORTH KOREA": {"code": "KP", "program": "DPRK", "risk": 1.0},
        "IRAN": {"code": "IR", "program": "IRAN", "risk": 1.0},
        "SYRIA": {"code": "SY", "program": "SYRIA", "risk": 1.0},
        "CUBA": {"code": "CU", "program": "CUBA", "risk": 1.0},
        "CRIMEA": {"code": "UA-CRI", "program": "UKRAINE-RUSSIA", "risk": 1.0},
    }
    
    # High-Risk Jurisdictions (FATF Grey/Black list)
    HIGH_RISK_COUNTRIES = {
        "MYANMAR": {"code": "MM", "risk": 0.8, "reason": "FATF Black List"},
        "PAKISTAN": {"code": "PK", "risk": 0.6, "reason": "FATF Grey List"},
        "NIGERIA": {"code": "NG", "risk": 0.5, "reason": "High fraud risk"},
        "SOMALIA": {"code": "SO", "risk": 0.7, "reason": "Terrorism financing"},
        "YEMEN": {"code": "YE", "risk": 0.7, "reason": "Conflict zone"},
        "VENEZUELA": {"code": "VE", "risk": 0.6, "reason": "Sectoral sanctions"},
        "RUSSIA": {"code": "RU", "risk": 0.7, "reason": "Ukraine-related sanctions"},
        "BELARUS": {"code": "BY", "risk": 0.7, "reason": "Ukraine-related sanctions"},
    }
    
    # High-Risk Merchant Category Codes (MCC)
    HIGH_RISK_MCC = {
        # Gambling
        "7995": {"category": "GAMBLING", "risk": 0.7, "description": "Betting/Casino/Lottery"},
        "7801": {"category": "GAMBLING", "risk": 0.7, "description": "Government Lottery"},
        "7802": {"category": "GAMBLING", "risk": 0.7, "description": "Horse/Dog Racing"},
        
        # Cryptocurrency
        "6051": {"category": "CRYPTO", "risk": 0.6, "description": "Cryptocurrency Exchange"},
        "6211": {"category": "CRYPTO", "risk": 0.5, "description": "Securities/Crypto Brokers"},
        
        # Money Services
        "6012": {"category": "MONEY_SERVICES", "risk": 0.6, "description": "Financial Institutions"},
        "6050": {"category": "MONEY_SERVICES", "risk": 0.7, "description": "Quasi Cash - Money Orders"},
        "6540": {"category": "MONEY_SERVICES", "risk": 0.7, "description": "Wire Transfer/Money Order"},
        
        # High-Value Goods (money laundering risk)
        "5944": {"category": "HIGH_VALUE", "risk": 0.5, "description": "Jewelry/Precious Metals"},
        "5571": {"category": "HIGH_VALUE", "risk": 0.5, "description": "Motorcycle Dealers"},
        "5511": {"category": "HIGH_VALUE", "risk": 0.4, "description": "Automobile Dealers"},
        
        # Adult/Digital
        "5967": {"category": "ADULT", "risk": 0.6, "description": "Direct Marketing - Adult"},
        "7273": {"category": "ADULT", "risk": 0.5, "description": "Dating Services"},
    }

    def __init__(self):
        """Initialize the OFAC screener."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for name matching."""
        self._name_patterns = []
        for name, aliases, entity_type, program in self.SDN_LIST:
            all_names = [name] + aliases
            for n in all_names:
                # Create pattern that matches the name with word boundaries
                pattern = re.compile(r'\b' + re.escape(n.upper()) + r'\b', re.IGNORECASE)
                self._name_patterns.append((pattern, name, entity_type, program))
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison."""
        # Remove special characters, extra spaces
        normalized = re.sub(r'[^\w\s]', '', name.upper())
        normalized = ' '.join(normalized.split())
        return normalized
    
    def screen_name(self, name: str) -> ScreeningResult:
        """
        Screen a name against the SDN list.
        
        Args:
            name: Name to screen (individual or entity)
            
        Returns:
            ScreeningResult with match details
        """
        if not name:
            return ScreeningResult(
                is_match=False,
                match_type="NO_MATCH",
                matched_entry=None,
                list_name="SDN",
                risk_score=0.0,
                reason_code="",
                description="Name is empty"
            )
        
        normalized_name = self._normalize_name(name)
        
        # Check exact matches first
        for pattern, sdn_name, entity_type, program in self._name_patterns:
            if pattern.search(name):
                return ScreeningResult(
                    is_match=True,
                    match_type="EXACT",
                    matched_entry=sdn_name,
                    list_name="SDN",
                    risk_score=1.0,
                    reason_code="OFAC_SDN_MATCH",
                    description=f"OFAC SDN Match: '{name}' matches sanctioned {entity_type.lower()} '{sdn_name}' under {program} program. Transaction MUST be blocked."
                )
        
        # Check fuzzy matches (partial name matches)
        for sdn_name, aliases, entity_type, program in self.SDN_LIST:
            sdn_parts = self._normalize_name(sdn_name).split()
            name_parts = normalized_name.split()
            
            # Check if last name matches
            if len(sdn_parts) >= 1 and len(name_parts) >= 1:
                if sdn_parts[0] == name_parts[-1]:  # Last name match (comma-separated format)
                    return ScreeningResult(
                        is_match=True,
                        match_type="FUZZY",
                        matched_entry=sdn_name,
                        list_name="SDN",
                        risk_score=0.7,
                        reason_code="OFAC_SDN_FUZZY",
                        description=f"Potential OFAC SDN Match: '{name}' may match '{sdn_name}' ({program}). Manual review required."
                    )
        
        return ScreeningResult(
            is_match=False,
            match_type="NO_MATCH",
            matched_entry=None,
            list_name="SDN",
            risk_score=0.0,
            reason_code="",
            description="No OFAC SDN match found"
        )
    
    def screen_country(self, country: str) -> ScreeningResult:
        """
        Screen a country against sanctions and high-risk lists.
        
        Args:
            country: Country name to screen
            
        Returns:
            ScreeningResult with risk details
        """
        if not country:
            return ScreeningResult(
                is_match=False,
                match_type="NO_MATCH",
                matched_entry=None,
                list_name="COUNTRY",
                risk_score=0.0,
                reason_code="",
                description="Country is empty"
            )
        
        country_upper = country.upper().strip()
        
        # Check fully sanctioned countries
        for sanctioned, details in self.SANCTIONED_COUNTRIES.items():
            if sanctioned in country_upper or country_upper in sanctioned:
                return ScreeningResult(
                    is_match=True,
                    match_type="EXACT",
                    matched_entry=sanctioned,
                    list_name="SANCTIONED_COUNTRY",
                    risk_score=1.0,
                    reason_code="SANCTIONED_COUNTRY",
                    description=f"SANCTIONED COUNTRY: '{country}' is under comprehensive {details['program']} sanctions. All transactions BLOCKED."
                )
        
        # Check high-risk jurisdictions
        for high_risk, details in self.HIGH_RISK_COUNTRIES.items():
            if high_risk in country_upper or country_upper in high_risk:
                return ScreeningResult(
                    is_match=True,
                    match_type="PARTIAL",
                    matched_entry=high_risk,
                    list_name="HIGH_RISK_COUNTRY",
                    risk_score=details["risk"],
                    reason_code="HIGH_RISK_COUNTRY",
                    description=f"HIGH-RISK JURISDICTION: '{country}' ({details['reason']}). Enhanced due diligence required."
                )
        
        return ScreeningResult(
            is_match=False,
            match_type="NO_MATCH",
            matched_entry=None,
            list_name="COUNTRY",
            risk_score=0.0,
            reason_code="",
            description="Country not on sanctions or high-risk list"
        )
    
    def screen_mcc(self, mcc_code: str) -> ScreeningResult:
        """
        Screen Merchant Category Code for high-risk categories.
        
        Args:
            mcc_code: 4-digit MCC code
            
        Returns:
            ScreeningResult with risk details
        """
        if not mcc_code:
            return ScreeningResult(
                is_match=False,
                match_type="NO_MATCH",
                matched_entry=None,
                list_name="MCC",
                risk_score=0.0,
                reason_code="",
                description="MCC code is empty"
            )
        
        mcc_str = str(mcc_code).strip()
        
        if mcc_str in self.HIGH_RISK_MCC:
            details = self.HIGH_RISK_MCC[mcc_str]
            return ScreeningResult(
                is_match=True,
                match_type="EXACT",
                matched_entry=mcc_str,
                list_name="HIGH_RISK_MCC",
                risk_score=details["risk"],
                reason_code=f"MCC_{details['category']}",
                description=f"HIGH-RISK MCC {mcc_str}: {details['description']} ({details['category']}). Enhanced monitoring required."
            )
        
        return ScreeningResult(
            is_match=False,
            match_type="NO_MATCH",
            matched_entry=None,
            list_name="MCC",
            risk_score=0.0,
            reason_code="",
            description="MCC not in high-risk category"
        )
    
    def screen_transaction(self, transaction: dict) -> list[ScreeningResult]:
        """
        Comprehensive transaction screening.
        
        Args:
            transaction: Transaction dict with user_name, country, mcc_code
            
        Returns:
            List of ScreeningResults for all checks
        """
        results = []
        
        # Screen user name if available
        user_name = transaction.get("user_name", "")
        if user_name:
            name_result = self.screen_name(user_name)
            if name_result.is_match:
                results.append(name_result)
        
        # Screen country/location
        country = transaction.get("country") or transaction.get("location", "")
        if country:
            country_result = self.screen_country(country)
            if country_result.is_match:
                results.append(country_result)
        
        # Screen MCC if available
        mcc_code = transaction.get("mcc_code", "")
        if mcc_code:
            mcc_result = self.screen_mcc(mcc_code)
            if mcc_result.is_match:
                results.append(mcc_result)
        
        return results


# Singleton instance
_ofac_screener = None


def get_ofac_screener() -> OFACScreener:
    """Get singleton OFAC screener instance."""
    global _ofac_screener
    if _ofac_screener is None:
        _ofac_screener = OFACScreener()
    return _ofac_screener


if __name__ == "__main__":
    # Demo screening
    screener = OFACScreener()
    
    print("=== OFAC SDN Screening Demo ===\n")
    
    # Test names
    test_names = [
        "Mohammad Ahmadi",
        "John Smith",
        "Alexei Petrov",
        "Golden Trade LLC"
    ]
    
    for name in test_names:
        result = screener.screen_name(name)
        status = "🚨 MATCH" if result.is_match else "✅ CLEAR"
        print(f"{status} | {name}")
        if result.is_match:
            print(f"   → {result.description}")
        print()
    
    print("=== Country Screening Demo ===\n")
    
    test_countries = ["Iran", "India", "Russia", "United States", "North Korea"]
    
    for country in test_countries:
        result = screener.screen_country(country)
        status = "🚨 BLOCKED" if result.risk_score >= 1.0 else ("⚠️ HIGH-RISK" if result.is_match else "✅ OK")
        print(f"{status} | {country}")
        if result.is_match:
            print(f"   → {result.description}")
        print()
    
    print("=== MCC Screening Demo ===\n")
    
    test_mccs = ["7995", "5411", "6051", "5944"]
    
    for mcc in test_mccs:
        result = screener.screen_mcc(mcc)
        status = "⚠️ HIGH-RISK" if result.is_match else "✅ OK"
        print(f"{status} | MCC {mcc}")
        if result.is_match:
            print(f"   → {result.description}")
        print()
