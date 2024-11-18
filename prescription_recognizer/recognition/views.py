from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.db.models import Q, Count, F, Value, FloatField
from django.urls import reverse
from .forms import PrescriptionForm
from .models import PrescriptionImage, Medicine
from PIL import Image
import torch
import re
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Tuple
from rapidfuzz import fuzz

# Initialize OCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

class PrescriptionProcessor:
    def __init__(self):
        self.section_indicators = {
            'header': ['dr.', 'doctor', 'clinic', 'hospital', 'medical'],
            'patient_info': ['name:', 'age:', 'gender:', 'date:'],
            'medicine': ['rx', 'r/x', 'medicine', 'prescribed'],
        }
        
        self.medicine_indicators = {
            'dosage_forms': [
                'tablet', 'tab', 'tabs', 'capsule', 'cap', 'caps', 'syrup',
                'injection', 'inj', 'drops', 'cream', 'ointment', 'suspension'
            ],
            'measurements': ['mg', 'ml', 'mcg', 'g', 'iu', '%'],
            'frequency': ['od', 'bid', 'tid', 'qid', 'hs', '1-0-1', '1-0-0', '0-0-1'],
            'duration': ['days', 'weeks', 'months', 'day', 'week', 'month']
        }

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better handwriting recognition."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Remove background noise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    def segment_prescription(self, image_path: str) -> Dict[str, Image.Image]:
        """Segment prescription using contour detection and text analysis."""
        # Read and enhance image
        img = cv2.imread(image_path)
        enhanced = self.enhance_image(img)
        
        # Find contours
        contours, _ = cv2.findContours(
            enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours by vertical position
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        # Initialize sections
        sections = {
            'header': None,
            'patient_info': None,
            'medications': None,
            'footer': None
        }
        
        height, width = enhanced.shape[:2]
        y_positions = []
        
        # Analyze each contour
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small contours
            if w < width * 0.1 or h < height * 0.02:
                continue
                
            section_img = enhanced[y:y+h, x:x+w]
            y_positions.append(y)
            
            # Convert to PIL for OCR
            pil_img = Image.fromarray(section_img)
            text = self.predict_text(pil_img).lower()
            
            # Classify section based on content and position
            if not sections['header'] and any(ind in text for ind in self.section_indicators['header']):
                sections['header'] = enhanced[0:y+h, :]
            elif not sections['patient_info'] and any(ind in text for ind in self.section_indicators['patient_info']):
                if sections['header']:
                    start_y = cv2.boundingRect(sorted_contours[0])[1]
                    sections['patient_info'] = enhanced[start_y:y+h, :]
            elif any(ind in text for ind in self.section_indicators['medicine']) or \
                 any(ind in text for indicators in self.medicine_indicators.values() for ind in indicators):
                medicine_start = y
                medicine_end = y + h
                # Include subsequent lines that might be part of medicine list
                for next_y in y_positions:
                    if next_y > y and next_y - (medicine_end) < height * 0.1:
                        medicine_end = next_y + h
                sections['medications'] = enhanced[medicine_start:medicine_end, :]
        
        # Convert sections to PIL Images
        return {
            section: Image.fromarray(img) if img is not None else None
            for section, img in sections.items()
        }

    def predict_text(self, image: Image.Image) -> str:
        """Enhanced OCR with better handling of handwritten text."""
        try:
            # Prepare image
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # Generate text with improved parameters
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                length_penalty=1.0
            )
            
            prediction = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            return prediction.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def extract_medicines(self, text: str) -> List[Dict[str, str]]:
        """Extract medicine information with improved pattern matching."""
        lines = text.split('\n')
        medicines = []
        current_medicine = None
        
        for line in lines:
            line = line.strip().lower()
            if not line:
                continue
            
            # Check if line starts new medicine entry
            if self._is_medicine_start(line):
                if current_medicine:
                    medicines.append(current_medicine)
                current_medicine = self._parse_medicine_line(line)
            elif current_medicine:
                # Update existing medicine entry with additional info
                self._update_medicine_info(current_medicine, line)
        
        if current_medicine:
            medicines.append(current_medicine)
        
        return medicines

    def _is_medicine_start(self, line: str) -> bool:
        """Check if line indicates start of new medicine entry."""
        patterns = [
            r'^\d+\.',  # Numbered list
            r'^[•\-*]',  # Bullet points
            r'^rx:?\s',  # Rx notation
            r'^\w+\s*\d+\s*(mg|ml|mcg)',  # Medicine with strength
            r'^tab\.|^cap\.',  # Common abbreviations
        ]
        return any(re.match(pattern, line) for pattern in patterns) or \
               any(form in line.split()[0] for form in self.medicine_indicators['dosage_forms'])

    def _parse_medicine_line(self, line: str) -> Dict[str, str]:
        """Parse medicine line with improved pattern recognition."""
        # Remove common prefixes
        line = re.sub(r'^(\d+\.|\-|•|rx:?\s)', '', line).strip()
        
        medicine = {
            'name': '',
            'strength': '',
            'dosage_form': '',
            'frequency': '',
            'duration': ''
        }
        
        # Extract medicine name and strength
        name_pattern = r'^([a-zA-Z\s-]+)(?:\s+(\d+(?:\.\d+)?(?:mg|ml|mcg|g|iu|%)))?'
        name_match = re.match(name_pattern, line)
        
        if name_match:
            medicine['name'] = name_match.group(1).strip()
            medicine['strength'] = name_match.group(2) if name_match.group(2) else ''
            
            # Process remaining text
            remaining = line[len(name_match.group(0)):].strip()
            self._update_medicine_info(medicine, remaining)
        
        return medicine

    def _update_medicine_info(self, medicine: Dict[str, str], text: str) -> None:
        """Update medicine information from additional text."""
        text = text.lower()
        
        # Extract dosage form
        for form in self.medicine_indicators['dosage_forms']:
            if form in text:
                medicine['dosage_form'] = form
                break
        
        # Extract frequency
        for freq in self.medicine_indicators['frequency']:
            if freq in text:
                medicine['frequency'] = freq
                break
        
        # Extract duration
        duration_match = re.search(r'(\d+)\s*(?:days?|weeks?|months?)', text)
        if duration_match:
            medicine['duration'] = duration_match.group(0)

    def find_matching_medicines(self, medicine_infos: List[Dict[str, str]], Medicine) -> List[Tuple[any, float]]:
        """Find matching medicines with improved fuzzy matching."""
        matches = []
        
        for med_info in medicine_infos:
            medicine_name = med_info['name']
            strength = med_info['strength']
            
            # Build query
            words = medicine_name.split()
            query = Q()
            
            for word in words:
                if len(word) > 2:
                    query |= (
                        Q(product_name__icontains=word) |
                        Q(salt_composition__icontains=word)
                    )
            
            potential_matches = Medicine.objects.filter(query)
            
            for medicine in potential_matches:
                # Calculate name similarity
                name_similarity = fuzz.ratio(
                    medicine_name.lower(),
                    medicine.product_name.lower()
                ) / 100.0
                
                # Calculate word match score
                word_match_score = sum(
                    1 for word in words
                    if word.lower() in medicine.product_name.lower() or
                    word.lower() in medicine.salt_composition.lower()
                ) / len(words)
                
                # Check strength match
                strength_score = 1.0 if strength and strength in medicine.product_name else 0.0
                
                # Calculate final score with weighted components
                final_score = (
                    name_similarity * 0.5 +
                    word_match_score * 0.3 +
                    strength_score * 0.2
                )
                
                if final_score > 0.3:
                    matches.append((medicine, final_score))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

def upload_prescription(request):
    """Handle prescription upload and processing."""
    if request.method == 'POST':
        form = PrescriptionForm(request.POST, request.FILES)
        if form.is_valid():
            prescription_image = form.save()
            
            # Process prescription
            processor = PrescriptionProcessor()
            sections = processor.segment_prescription(prescription_image.image.path)
            
            # Focus on medications section
            if sections['medications']:
                medications_text = processor.predict_text(sections['medications'])
                medicine_infos = processor.extract_medicines(medications_text)
                matching_medicines = processor.find_matching_medicines(
                    medicine_infos, Medicine
                )
                
                prescription_image.processed = True
                prescription_image.save()
                
                return render(request, 'recognition/results.html', {
                    'prescription': prescription_image,
                    'medicine_infos': medicine_infos,
                    'matching_medicines': matching_medicines
                })
    else:
        form = PrescriptionForm()
    
    return render(request, 'recognition/upload.html', {'form': form})

def medicine_detail(request, id):
    """Display medicine details."""
    medicine = get_object_or_404(Medicine, id=id)
    return render(request, 'recognition/medicine_detail.html', {'medicine': medicine})