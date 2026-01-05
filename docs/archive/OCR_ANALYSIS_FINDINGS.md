# OCR Deep Analysis - Findings & Recommendations

**Date**: January 5, 2026
**Test Dataset**: 1,195 samples (195 message boards, 500 arrow boards, 500 TTC signs)

---

## EXECUTIVE SUMMARY

### Detection Performance
- **Overall Detection Rate**: 24.0% (287/1,195)
  - Message boards: 30.3% (59/195) ✅ Best
  - TTC signs: 34.0% (170/500)
  - Arrow boards: 11.6% (58/500) ❌ Worst

### Quality of Detections
- **OCR confidence (when detected)**: 0.87 avg ✅ HIGH
- **Fused confidence**: 0.35 avg ❌ LOW
- **Problem**: 73.9% classified as "UNCLEAR" (212/287)

---

## CRITICAL FINDINGS

### ✅ What Works Well (75 high-quality detections)

**WORKZONE (19 detections - 100% valid)**
- "ROAD WORK AHEAD" (x9)
- "UTILITY WORK AHEAD" (x3)
- "WORK ZONE" (x2)
- "CONSTRUCTION TRAFFIC ONLY"
- "CONSTRUCTION ENTRANCE AHEAD"
- Avg confidence: 0.899 ✅

**LANE (10 detections - 100% valid)**
- "LANE CLOSED" (x5)
- "RIGHT LANE CLOSED" (x2)
- "LEFT LANE CLOSED AHEAD"
- "LANE SHIFT"
- Avg confidence: 0.894 ✅

**DIRECTION (39 detections - 90% valid)**
- "DETOUR" (x13)
- "SHIFT AHEAD" (x3)
- "FOLLOW DETOUR SIDEWALK CLOSED" (x2)
- "LANES SHIFT AHEAD" (x3)
- "ROAD CLOSED AHEAD"
- Avg confidence: 0.851 ✅

**CAUTION (6 detections - 100% valid)**
- "CAUTION" (x3)
- "KEEP ALERT"
- Avg confidence: 0.856 ✅

**SPEED (1 detection - 100% valid)**
- "WORKZONE SPEED 45 MPH"
- Avg confidence: 0.834 ✅

**Total high-quality**: 75/287 = **26% of detections are reliable**

### ❌ What's Broken (212 UNCLEAR detections)

**Major Issue**: Classifier too restrictive, missing obvious patterns

**Category 1: ROAD CLOSED variants (47 instances)**
- "ROAD CLOSED" (x14)
- "STREET CLOSED", "SIDEWALK CLOSED", "RAMP CLOSED"
- "ROAD CLOSED TO THRU TRAFFIC" (x2)
- **Should be**: LANE or DIRECTION category

**Category 2: Short/Noise (30+ instances)**
- Single letters: "M", "W", "N", "E", "T"
- Numbers: "36", "'37", "510", "70", "22"
- Codes: "FM20", "HN", "AF2", "632 T.$.1."
- **Should be**: FILTERED OUT

**Category 3: Company names/irrelevant (20+ instances)**
- "ARROWBOARDS.COM" (x3)
- "RoadSafe 412-767-98.10"
- "TMI TRAFFIC MANAGEMENT"
- "JOSEPH B. FAY COMPANY"
- **Should be**: FILTERED OUT

**Category 4: Misclassified workzone text (28 instances)**
- "MOOR ZONE" (typo of WORK ZONE)
- "ROAD HORK AHEAC" (typo of ROAD WORK AHEAD)
- **Should be**: WORKZONE

**Category 5: Valid directional text (12+ instances)**
- "DO NOT ENTER" (x2)
- "ONE WAY"
- "NO TRUCK TURN AT"
- **Should be**: DIRECTION

---

## ROOT CAUSES

### 1. **Classifier Patterns Too Specific**
Current patterns miss common variations:
- ❌ Missing: "ROAD CLOSED" (not in any pattern)
- ❌ Missing: "SIDEWALK CLOSED" 
- ❌ Missing: "RAMP CLOSED"
- ❌ Missing: "DO NOT ENTER"
- ❌ Missing: "ONE WAY"

### 2. **No Noise Filtering**
OCR returns garbage but classifier tries to categorize:
- Single letters/numbers should be rejected
- Text < 3 characters should be filtered
- Non-English gibberish should be filtered

### 3. **No Fuzzy Matching**
Typos common in OCR but patterns require exact match:
- "MOOR ZONE" vs "WORK ZONE"
- "HORK" vs "WORK"
- "CLDSED" vs "CLOSED"

### 4. **Missing Common Workzone Vocabulary**
Patterns don't cover real-world signs:
- "ROAD CLOSED" (appears 14x!)
- "DETOUR" (appears 13x but only in DIRECTION)
- "TRAFFIC MANAGEMENT"
- "UTILITY WORK"

---

## RECOMMENDED FIXES

### Priority 1: Expand Classifier Patterns

**WORKZONE additions**:
```python
r'road\s*closed',    # NEW: appears 14x
r'utility',          # NEW: "UTILITY WORK AHEAD"
r'construction',     # Already have, keep
r'traffic\s*only',   # NEW: "CONSTRUCTION TRAFFIC ONLY"
```

**LANE additions**:
```python
r'closed',           # NEW: generic "closed" for lanes/ramps
r'sidewalk',         # NEW: sidewalk closures
r'ramp\s*closed',    # NEW: ramp closures
r'street\s*closed',  # NEW: street closures
r'shift',            # Already have "lane shift"
```

**DIRECTION additions**:
```python
r'do\s*not\s*enter', # NEW: appears 2x
r'one\s*way',        # NEW: appears 1x
r'turn',             # NEW: "NO TURN", "RIGHT TURN"
r'enter',            # Already have in other contexts
r'exit',             # Already have
```

### Priority 2: Add Noise Filter

**Pre-classification filter**:
```python
def is_noise(text: str) -> bool:
    """Filter out OCR garbage"""
    text = text.strip()
    
    # Too short
    if len(text) < 3:
        return True
    
    # Single character repeated
    if len(set(text.replace(' ', ''))) == 1:
        return True
    
    # No letters (numbers/symbols only)
    if not re.search(r'[A-Za-z]{2,}', text):
        return True
    
    # Known company names/websites
    noise_keywords = ['arrowboards.com', 'roadsafe', '.com', 
                      'joseph', 'fay', 'traffic management']
    if any(k in text.lower() for k in noise_keywords):
        return True
    
    return False
```

### Priority 3: Fuzzy Matching (Optional)

Use Levenshtein distance for common typos:
- "HORK" → "WORK" (distance=1)
- "CLDSED" → "CLOSED" (distance=1)
- "MOOR" → "WORK" (distance=2)

---

## REVISED PERFORMANCE ESTIMATES

**After fixes**:
- UNCLEAR: 212 → **~50** (filter noise + expand patterns)
- WORKZONE: 19 → **~45** (add "ROAD CLOSED" variants)
- LANE: 10 → **~55** (add closure patterns)
- DIRECTION: 39 → **~50** (add "DO NOT ENTER", etc)
- **Filtered noise**: ~50-60 detections removed
- **Net usable detections**: 75 → **~200/287** (70% quality vs 26%)

---

## INTEGRATION RECOMMENDATION

**Based on improved classifier**:
- ✅ **Use OCR with weight 0.15-0.20** (moderate contribution)
- ✅ **Minimum confidence threshold: 0.7** (high-quality only)
- ✅ **Focus categories**: WORKZONE, LANE, DIRECTION (ignore UNCLEAR)
- ✅ **Frame sampling**: 1 Hz (every 30 frames) for Jetson
- ✅ **Expected accuracy boost**: +8-12% on state machine

