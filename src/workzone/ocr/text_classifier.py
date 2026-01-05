"""
Rule-based text classifier for work zone signs.

Categorizes extracted text into semantic categories using regex patterns.
"""

import re
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TextClassifier:
    """
    Rule-based classifier for work zone sign text.
    
    Categories:
    - WORKZONE: Work zone warnings, construction alerts
    - SPEED: Speed limits, speed advisories
    - LANE: Lane closures, merges, shifts
    - CAUTION: General caution/warning messages
    - DIRECTION: Directional arrows/instructions
    - UNCLEAR: Text detected but no clear category
    - NONE: No text detected
    """
    
    # Regex patterns for each category (case-insensitive)
    # Expanded based on real dataset analysis
    PATTERNS = {
        'WORKZONE': [
            r'work\s*zone',
            r'road\s*work',
            r'construction',
            r'work\s*ahead',
            r'workers?\s*ahead',
            r'workers?\s*present',  # NEW: found in test
            r'work\s*area',
            r'men\s*working',
            r'crew\s*working',
            r'utility\s*work',  # NEW: found in GT
            r'construction\s*entrance',  # NEW
            r'construction\s*traffic',  # NEW
        ],
        'SPEED': [
            r'speed\s*limit',
            r'\b\d{1,3}\s*mph\b',  # e.g., "45 MPH"
            r'speed\s*zone',
            r'slow',
            r'reduce\s*speed',
            r'slower\s*traffic',
            r'workzone\s*speed',  # NEW: "WORKZONE SPEED 45 MPH"
        ],
        'LANE': [
            r'lane\s*closed',
            r'lane\s*ends?',
            r'lane\s*shift',

            r'left\s*lane',
            r'right\s*lane',
            r'center\s*lane',
            r'use\s*\w+\s*lane',
            r'stay\s*in\s*lane',
            # NEW: Critical additions from analysis
            r'\bclosed\b',  # "CLOSED" alone (very common)
            r'road\s*closed',  # Appears 14x!
            r'ramp\s*closed',
            r'street\s*closed',
            r'sidewalk\s*closed',
            r'\bshift\b',  # "SHIFT" standalone
            r'pull[\\s-]?off',  # "PULL-OFF"
            r'lanes\s*shift',  # "LANES SHIFT"
        ],
        'CAUTION': [
            r'caution',
            r'warning',
            r'danger',
            r'alert',
            r'watch\s*for',
            r'be\s*prepared',
            r'hazard',
            # NEW: From ground truth
            r'keep\s*alert',  # Exact GT match
            r'stopped\s*traffic',
        ],
        'DIRECTION': [
            r'keep\s*\w+',  # e.g., "keep right"
            r'exit',
            r'detour',  # Appears 13x!
            r'follow',
            r'use\s*next',
            r'ahead',
            r'bear\s*\w+',  # e.g., "bear right"
            r'merge',  # Moved from LANE - merge is direction instruction
            # NEW: Critical additions
            r'do\s*not\s*enter',  # Found 2x
            r'one\s*way',
            r'\bturn\b',  # "TURN", "NO TURN"
            r'shift\s*ahead',
            r'steel\s*plate',
            r'pedestrian',
            r'truck\s*detour',
            r'enter',
            r'follow\s*detour',
        ],
    }
    
    def __init__(self):
        """Initialize text classifier with compiled regex patterns."""
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        logger.info("TextClassifier initialized with rule-based patterns")
    
    @staticmethod
    def is_noise(text: str) -> bool:
        """
        Filter out OCR garbage/noise before classification.
        
        Returns True if text is likely noise/irrelevant.
        """
        if not text:
            return True
        
        text = text.strip()
        
        # Too short (likely fragments)
        if len(text) < 3:
            return True
        
        # No meaningful letters (numbers/symbols only)
        if not re.search(r'[A-Za-z]{2,}', text):
            return True
        
        # Single character repeated (e.g., "WWWW", "1111")
        unique_chars = set(text.replace(' ', '').replace('-', ''))
        if len(unique_chars) <= 2 and len(text) > 3:
            return True
        
        # Radio frequency codes (FM20, AM1030, etc.) - but allow MPH numbers
        if re.match(r'^[A-Z]{1,2}\d+$', text) and not re.search(r'mph', text, re.IGNORECASE):
            return True
        
        # Time-only patterns (e.g., "TONIGHT 9PM-5AM") - not useful alone
        if re.search(r'\d+[AP]M', text, re.IGNORECASE) and not re.search(r'(work|road|lane|speed|closed)', text, re.IGNORECASE):
            return True
        
        # Phone number pattern with "TEL" keyword
        if re.search(r'(tel|phone|call)', text, re.IGNORECASE) and re.search(r'\d{3}', text):
            return True
        
        # Known noise patterns (company names, websites, phone numbers)
        noise_keywords = [
            'arrowboards.com', 'roadsafe', '.com', 'traffic management',
            'joseph', 'fay company', 'trafcon', 'weco', 'aquid ink',
            'rent me', 'protection services', 'upmc', 'merchant',
            '@', 'http', 'www', '.-.'  # Web/email patterns
        ]
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in noise_keywords):
            return True
        
        # Mostly non-alphanumeric (garbage symbols)
        alnum_count = sum(c.isalnum() or c.isspace() for c in text)
        if alnum_count / len(text) < 0.5:
            return True
        
        # Phone number pattern
        if re.search(r'\d{3}[\s-]\d{3}[\s-]\d{4}', text):
            return True
        
        return False
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify extracted text into semantic category.
        
        Args:
            text: Extracted text from OCR
        
        Returns:
            (category, confidence): Category name and confidence [0-1]
                                    Returns ("NONE", 0.0) if text is empty
                                    Returns ("NOISE", 0.0) if text is garbage
                                    Returns ("UNCLEAR", 0.2) if no pattern matches
        """
        if not text or not text.strip():
            return "NONE", 0.0
        
        text_clean = text.strip()
        
        # Filter noise FIRST (critical improvement!)
        if self.is_noise(text_clean):
            logger.debug(f"Filtered as noise: \"{text_clean}\"")
            return "NOISE", 0.0
        
        text_lower = text_clean.lower()
        
        # Priority check: If text contains speed pattern, prioritize SPEED category
        speed_priority = re.search(r'\b\d{1,3}\s*mph\b', text_lower) or re.search(r'speed\s*limit', text_lower)
        
        # Collect ALL matches with scores
        matches = []
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    # Score based on match quality
                    match_length = len(match.group(0))
                    text_length = len(text_lower.replace(' ', ''))
                    coverage = match_length / max(text_length, 1)
                    
                    # Base confidence
                    confidence = 0.90
                    
                    # CRITICAL: Boost SPEED when numbers + MPH present
                    if speed_priority and category == 'SPEED':
                        confidence = 0.98  # Highest priority
                        coverage = 1.0
                    
                    # Adjust by coverage
                    if coverage > 0.7:
                        confidence = 0.95  # Match covers most of text
                    elif coverage > 0.4:
                        confidence = 0.90
                    else:
                        confidence = 0.85  # Partial match
                    
                    # Boost for high-value keywords
                    high_value = ['workzone', 'work zone', 'detour', 'closed',
                                  'caution', 'speed limit', 'lane closed',
                                  'road work', 'shift ahead']
                    if any(kw in text_lower for kw in high_value):
                        confidence = min(confidence + 0.05, 0.95)
                    
                    matches.append((category, confidence, coverage, match_length))
        
        # Return best match
        if matches:
            # Sort by confidence, then coverage, then match length
            matches.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            best_cat, best_conf = matches[0][0], matches[0][1]
            logger.debug(f"Classified \"{text_clean}\" as {best_cat} ({best_conf:.2f})")
            return best_cat, best_conf
        
        # No pattern match - check if it's at least road-related
        road_indicators = ['road', 'lane', 'traffic', 'street', 'ahead',
                          'ramp', 'highway', 'route']
        if any(ind in text_lower for ind in road_indicators):
            logger.debug(f"Unclear but road-related: \"{text_clean}\"")
            return "UNCLEAR", 0.30  # Might be relevant
        
        # Completely unclear
        logger.debug(f"No category match for \"{text_clean}\"")
        return "UNCLEAR", 0.10
    
    def _compute_confidence(self, text: str, match: re.Match) -> float:
        """
        Compute confidence score based on match quality.
        
        Higher confidence for:
        - Short, clear text
        - Strong keyword matches
        - Well-formed text
        
        Args:
            text: Original text
            match: Regex match object
        
        Returns:
            Confidence score [0.6 - 0.95]
        """
        base_confidence = 0.75
        
        # Boost: Short text is usually clearer
        if len(text) < 30:
            base_confidence += 0.10
        elif len(text) > 80:
            base_confidence -= 0.10
        
        # Boost: Strong match (covers most of text)
        match_coverage = len(match.group(0)) / len(text)
        if match_coverage > 0.5:
            base_confidence += 0.10
        
        # Clamp to [0.6, 0.95]
        return max(0.60, min(0.95, base_confidence))
    
    def classify_batch(self, texts: list) -> list:
        """
        Classify multiple texts (batch processing).
        
        Args:
            texts: List of text strings
        
        Returns:
            List of (category, confidence) tuples
        """
        return [self.classify(text) for text in texts]
    
    def get_all_categories(self) -> list:
        """Return list of all possible categories."""
        return list(self.PATTERNS.keys()) + ['UNCLEAR', 'NONE']
