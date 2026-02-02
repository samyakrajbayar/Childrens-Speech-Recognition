"""
Whisper Text Normalizer
Standardizes ASR output for consistent scoring.
"""
import re
import unicodedata
from typing import Optional


class WhisperNormalizer:
    """Text normalizer following Whisper's normalization rules"""
    
    def __init__(self):
        # Common contractions mapping
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
        }
        
        # Number words
        self.number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
        }
    
    def __call__(self, text: str) -> str:
        """Normalize text"""
        return self.normalize(text)
    
    def normalize(self, text: str) -> str:
        """Full normalization pipeline"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Remove diacritics
        text = self._remove_diacritics(text)
        
        # Expand contractions
        text = self._expand_contractions(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Remove punctuation (keep apostrophes for now)
        text = self._remove_punctuation(text)
        
        # Normalize numbers (small numbers to words)
        text = self._normalize_numbers(text)
        
        # Final whitespace cleanup
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks"""
        # Decompose unicode characters
        decomposed = unicodedata.normalize("NFD", text)
        # Remove combining marks
        without_diacritics = "".join(
            char for char in decomposed 
            if unicodedata.category(char) != "Mn"
        )
        return without_diacritics
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces"""
        # Replace various whitespace with space
        text = re.sub(r'[\s\u00a0\u2000-\u200b\u202f\u205f\u3000]+', ' ', text)
        return text.strip()
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation except apostrophes"""
        # Keep letters, numbers, spaces, and apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        # Remove standalone apostrophes
        text = re.sub(r"\s'\s", ' ', text)
        text = re.sub(r"^'|'$", '', text)
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert small numbers to words"""
        words = text.split()
        normalized = []
        
        for word in words:
            if word in self.number_words:
                normalized.append(self.number_words[word])
            else:
                normalized.append(word)
        
        return ' '.join(normalized)


class EnhancedNormalizer(WhisperNormalizer):
    """Enhanced normalizer with additional rules for children's speech"""
    
    def __init__(self):
        super().__init__()
        
        # Common children's speech patterns
        self.child_patterns = {
            r'\bwanna\b': 'want to',
            r'\bgonna\b': 'going to',
            r'\bgotta\b': 'got to',
            r'\blemme\b': 'let me',
            r'\bkinda\b': 'kind of',
            r'\bsorta\b': 'sort of',
            r'\bdunno\b': 'do not know',
            r'\bcuz\b': 'because',
            r'\bcause\b': 'because',
            r'\byeah\b': 'yes',
            r'\bnope\b': 'no',
            r'\byep\b': 'yes',
            r'\buh huh\b': 'yes',
            r'\buh-huh\b': 'yes',
            r'\bnuh uh\b': 'no',
            r'\bnuh-uh\b': 'no',
        }
        
        # Filler words to optionally remove
        self.fillers = ['um', 'uh', 'hmm', 'hm', 'ah', 'er', 'eh']
    
    def normalize(self, text: str, remove_fillers: bool = False) -> str:
        """Enhanced normalization for children's speech"""
        text = super().normalize(text)
        
        # Apply child-specific patterns
        for pattern, replacement in self.child_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Optionally remove filler words
        if remove_fillers:
            text = self._remove_fillers(text)
        
        # Final cleanup
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _remove_fillers(self, text: str) -> str:
        """Remove filler words"""
        words = text.split()
        return ' '.join(w for w in words if w not in self.fillers)


def normalize_for_wer(hypothesis: str, reference: Optional[str] = None) -> str:
    """
    Normalize text for Word Error Rate calculation.
    Uses the enhanced normalizer for children's speech.
    """
    normalizer = EnhancedNormalizer()
    return normalizer.normalize(hypothesis)
