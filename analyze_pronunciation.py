import json
import logging
from typing import Dict, List, Any, Tuple
import speech_recognition as sr
from transcription import Transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PronunciationAnalyzer:
    def __init__(self):
        self.transcriber = Transcriber()
        # Common word to phoneme mappings (simplified CMU dict style)
        self.word_phonemes = {
            'we': ['W', 'IY'],
            'decided': ['D', 'IH', 'S', 'AY', 'D', 'IH', 'D'],
            'to': ['T', 'UW'],
            'go': ['G', 'OW'],
            'the': ['DH', 'AH'],
            'pool': ['P', 'UW', 'L'],
            'because': ['B', 'IH', 'K', 'AO', 'Z'],
            'it': ['IH', 'T'],
            'was': ['W', 'AH', 'Z'],
            'hot': ['HH', 'AA', 'T'],
            'outside': ['AW', 'T', 'S', 'AY', 'D']
        }
        
    def load_utterance_scores(self, utterance_file: str) -> Dict:
        """Load utterance scores from JSON file."""
        with open(utterance_file, 'r') as f:
            return json.load(f)

    def analyze_pronunciation(self, audio_file: str, utterance_file: str) -> Dict:
        """Analyze pronunciation by combining transcription and utterance scores."""
        # Get transcription with timestamps
        timestamped_result, transcript = self.transcriber.transcribe_audio(audio_file)
        
        # Load utterance scores
        utterance_scores = self.load_utterance_scores(utterance_file)
        
        # Get the full phoneme sequence and scores
        phonemes = utterance_scores.get("phonemes", "").split()
        phone_scores = utterance_scores.get("phone_scores", [])
        phone_durations = utterance_scores.get("phone_durations", [])
        
        # Initialize result structure
        result = {
            "status": "success",
            "speech_score": {
                "transcript": transcript,
                "word_score_list": []
            }
        }
        
        # Create a list of words from the transcript
        words = transcript.split()
        
        # Process each word and map phonemes
        current_phone_idx = 0
        for word in words:
            word = word.lower()  # Normalize to lowercase
            expected_phones = self.word_phonemes.get(word, [])
            
            # Find the phonemes for this word
            word_score = self._analyze_word(
                word, 
                current_phone_idx, 
                expected_phones,
                phone_scores, 
                phone_durations
            )
            result["speech_score"]["word_score_list"].append(word_score)
            
            # Update the phoneme index based on expected phoneme count
            if expected_phones:
                current_phone_idx += len(expected_phones)
            else:
                # If word not in dictionary, try to estimate number of phonemes
                current_phone_idx += len(word)  # Rough estimate
        
        return result
    
    def _analyze_word(self, word: str, start_idx: int, expected_phones: List[str],
                     phone_scores: List, phone_durations: List) -> Dict:
        """Analyze a single word's pronunciation."""
        word_score = {
            "word": word,
            "quality_score": 0,
            "phone_score_list": [],
            "syllable_score_list": []
        }
        
        # Process expected number of phones for this word
        phone_list = []
        for i, expected_phone in enumerate(expected_phones):
            idx = start_idx + i
            if idx < len(phone_scores):
                score = phone_scores[idx]
                duration = phone_durations[idx] if idx < len(phone_durations) else None
                
                if duration and score[1] != 'SIL':  # Skip silence markers
                    phone_list.append({
                        "phone": score[1],
                        "stress_level": None,
                        "extent": [
                            int(float(duration[1]) * 1000),  # Convert to milliseconds
                            int(float(duration[2]) * 1000)
                        ],
                        "quality_score": float(score[3]) * 100,  # Convert to percentage
                        "sound_most_like": score[1]
                    })
        
        if phone_list:
            word_score["phone_score_list"] = phone_list
            # Calculate average quality score for the word
            word_score["quality_score"] = sum(p["quality_score"] for p in phone_list) / len(phone_list)
            
            # Create syllable analysis
            syllables = self._detect_syllables(word, phone_list)
            word_score["syllable_score_list"] = syllables
        
        return word_score
    
    def _detect_syllables(self, word: str, phones: List[Dict]) -> List[Dict]:
        """Detect syllables in a word based on its phones."""
        # Basic syllable detection - one syllable per vowel sound
        vowel_sounds = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
        syllables = []
        current_syllable = []
        syllable_phones = 0
        
        for phone in phones:
            current_syllable.append(phone)
            syllable_phones += 1
            
            # If we find a vowel sound, mark it as end of syllable
            if phone['phone'] in vowel_sounds:
                if current_syllable:
                    start_time = current_syllable[0]['extent'][0]
                    end_time = current_syllable[-1]['extent'][1]
                    quality_scores = [p['quality_score'] for p in current_syllable]
                    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                    
                    syllables.append({
                        "phone_count": syllable_phones,
                        "stress_level": 1 if len(syllables) == 0 else 0,  # Primary stress on first syllable
                        "letters": word,  # This is simplified - ideally would split word into syllables
                        "quality_score": avg_quality,
                        "stress_score": 100,
                        "extent": [start_time, end_time]
                    })
                    current_syllable = []
                    syllable_phones = 0
        
        # Add any remaining phones as part of the last syllable
        if current_syllable:
            start_time = current_syllable[0]['extent'][0]
            end_time = current_syllable[-1]['extent'][1]
            quality_scores = [p['quality_score'] for p in current_syllable]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            syllables.append({
                "phone_count": syllable_phones,
                "stress_level": 0,
                "letters": word,
                "quality_score": avg_quality,
                "stress_score": 100,
                "extent": [start_time, end_time]
            })
        
        return syllables if syllables else [{
            "phone_count": len(phones),
            "stress_level": 1,
            "letters": word,
            "quality_score": sum(p["quality_score"] for p in phones) / len(phones) if phones else 0,
            "stress_score": 100,
            "extent": [phones[0]['extent'][0], phones[-1]['extent'][1]] if phones else [0, 0]
        }]

def main():
    analyzer = PronunciationAnalyzer()
    result = analyzer.analyze_pronunciation(
        "2_converted.wav",
        "utterance_score.json"
    )
    
    # Save results
    with open("analysis_output.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logging.info("Analysis completed and saved to analysis_output.json")

if __name__ == "__main__":
    main()