import streamlit as st
import os
import time
import re
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
from urllib.parse import urlparse, parse_qs, quote
import tempfile
import io
import base64
from pathlib import Path
import numpy as np
import shutil
from typing import List, Dict, Any, Tuple

# Replit Environment Detection and Configuration
IS_REPLIT = os.getenv('REPLIT_ENVIRONMENT') is not None or os.getenv('REPL_SLUG') is not None

if IS_REPLIT:
    st.set_page_config(
        page_title="Health Information Assistant - Replit",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Show Replit-specific welcome message
    if 'replit_welcome_shown' not in st.session_state:
        st.success("🚀 Running on Replit! Welcome to the Health Information Assistant.")
        st.info("💡 Tip: Add your API keys in the sidebar to get started.")
        st.session_state['replit_welcome_shown'] = True

# Add Google API client imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    youtube_api_support = True
except ImportError:
    youtube_api_support = False

# Add Pinecone imports
try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    pinecone_support = True
except ImportError:
    pinecone_support = False

# Add PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    pdf_support = True
except ImportError:
    pdf_support = False

try:
    import pytesseract
    from pdf2image import convert_from_path
    ocr_support = True
except ImportError:
    ocr_support = False

def get_available_models(api_key):
    """Detect which models the API key has access to with enhanced o3-mini support"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        
        if response.status_code == 200:
            models = response.json()["data"]
            model_ids = [m["id"] for m in models]
            
            # Debug logging for o3-mini detection
            st.write("🔍 **Debug: Available models from API:**", model_ids[:10] + ["..."] if len(model_ids) > 10 else model_ids)
            
            # Enhanced model mapping for better detection
            model_mapping = {
                "gpt-4o": None,
                "gpt-4": None,
                "gpt-3.5-turbo": None,
                "o3-mini": None,
                "o1-preview": None,
                "o1": None,
                "gpt-4-turbo": None
            }
            
            # Find exact matches or partial matches
            for model_id in model_ids:
                model_lower = model_id.lower()
                
                # Check for o3-mini variants (e.g., "o3-mini-2025-01-31")
                if "o3-mini" in model_lower:
                    model_mapping["o3-mini"] = model_id
                    st.success(f"✅ Found o3-mini: {model_id}")
                elif "gpt-4o" in model_lower and not model_mapping["gpt-4o"]:
                    model_mapping["gpt-4o"] = model_id
                elif model_id.startswith("gpt-4") and "turbo" in model_lower and not model_mapping["gpt-4-turbo"]:
                    model_mapping["gpt-4-turbo"] = model_id
                elif model_id.startswith("gpt-4") and not model_mapping["gpt-4"]:
                    model_mapping["gpt-4"] = model_id
                elif "gpt-3.5-turbo" in model_lower and not model_mapping["gpt-3.5-turbo"]:
                    model_mapping["gpt-3.5-turbo"] = model_id
                elif "o1-preview" in model_lower:
                    model_mapping["o1-preview"] = model_id
                elif model_id.startswith("o1") and not model_mapping["o1"]:
                    model_mapping["o1"] = model_id
            
            # Return available models with their actual IDs
            available = []
            for base_name, actual_id in model_mapping.items():
                if actual_id:
                    available.append((base_name, actual_id))
            
            # Debug what was found
            st.write("🎯 **Found models:**", dict(available))
            
            # Return just the base names for UI, but store full mapping in session
            st.session_state['model_mapping'] = dict(available)
            return [base_name for base_name, _ in available] if available else ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
            
    except Exception as e:
        st.error(f"❌ Error detecting models: {str(e)}")
        return ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]  # Default fallback

def prepare_api_payload(model, messages, temperature=0.1, max_tokens=2000, **kwargs):
    """
    Prepare API payload with proper parameter handling for different model types.
    O3 models don't support temperature and use max_completion_tokens instead of max_tokens.
    """
    # Get the actual model ID from session mapping if available
    model_mapping = st.session_state.get('model_mapping', {})
    actual_model_id = model_mapping.get(model, model)
    
    payload = {
        "model": actual_model_id,
        "messages": messages
    }
    
    # Handle o3 models which have different parameter requirements
    if "o3" in actual_model_id.lower():
        # O3 models don't support temperature and use max_completion_tokens
        payload["max_completion_tokens"] = max_tokens
        if "reasoning_effort" not in kwargs:
            payload["reasoning_effort"] = "medium"  # Default reasoning effort
    else:
        # Regular models support temperature and use max_tokens
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    
    # Add any additional parameters
    payload.update(kwargs)
    
    return payload

def make_openai_request(payload, api_key, endpoint="chat/completions"):
    """
    Make OpenAI API request with comprehensive error handling for o3-mini
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(
            f"https://api.openai.com/v1/{endpoint}",
            headers=headers,
            json=payload,
            timeout=120  # Longer timeout for o3 models
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_details = response.text
            error_msg = f"API Error {response.status_code}: {error_details}"
            
            # Specific error handling for o3-mini
            if "o3" in payload.get("model", "").lower():
                if response.status_code == 400:
                    if "invalid model" in error_details.lower():
                        error_msg = "❌ o3-mini model not available with your API key. Please check your OpenAI subscription tier."
                    elif "temperature" in error_details.lower():
                        error_msg = "❌ o3-mini doesn't support temperature parameter. Using default settings."
                elif response.status_code == 429:
                    error_msg = "❌ Rate limit exceeded for o3-mini. Please wait a moment and try again."
                elif response.status_code == 403:
                    error_msg = "❌ Access denied to o3-mini. This model requires higher API usage tiers."
            
            return {"success": False, "error": error_msg, "status_code": response.status_code}
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. o3-mini responses can take longer due to reasoning."}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# Model-specific configurations
MODEL_CONFIGS = {
    "gpt-4o": {
        "temperature": 0.1, 
        "max_tokens": 2000,
        "description": "Best for medical analysis"
    },
    "gpt-4": {
        "temperature": 0.1, 
        "max_tokens": 2000,
        "description": "High quality, slower"
    },
    "gpt-3.5-turbo": {
        "temperature": 0.1, 
        "max_tokens": 1500,
        "description": "Faster, lower cost"
    },
    "o3-mini": {
        "temperature": 0.0,  # Not used for o3 models but kept for consistency
        "max_tokens": 2500,
        "description": "O3 series - enhanced reasoning",
        "reasoning_effort": "medium",  # o3-specific parameter
        "special_handling": True  # Flag for special API handling
    },
    "o1-preview": {
        "temperature": 0.1, 
        "max_tokens": 2000,
        "description": "O1 series preview"
    },
    "o1": {
        "temperature": 0.1,
        "max_tokens": 2000,
        "description": "O1 series reasoning"
    },
    "gpt-4-turbo": {
        "temperature": 0.1,
        "max_tokens": 2000,
        "description": "GPT-4 Turbo - fast and capable"
    }
}

# Set page configuration (only if not on Replit)
if not IS_REPLIT:
    st.set_page_config(
        page_title="Health Claim Analyzer",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Global variables
HISTORY_FILE = "analysis_history.csv"
GUIDELINES_DIR = "society_guidelines"
GUIDELINES_INDEX = "guidelines_index.json"
GUIDELINES_PDF_DIR = os.path.join(GUIDELINES_DIR, "pdf")
GUIDELINES_TEXT_DIR = os.path.join(GUIDELINES_DIR, "text")
GUIDELINES_TEMP_DIR = os.path.join(GUIDELINES_DIR, "temp")

# Pinecone configuration
PINECONE_INDEX_NAME = "health-guidelines"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection"""
    if not pinecone_support:
        return None
    
    pinecone_api_key = st.session_state.get('pinecone_api_key')
    if not pinecone_api_key:
        return None
    
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            # Create index if it doesn't exist
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            time.sleep(10)  # Wait for index to be ready
        
        index = pc.Index(PINECONE_INDEX_NAME)
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

# Embedding functions
def get_embedding(text: str, openai_api_key: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Generate embedding for text using OpenAI API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    payload = {
        "model": model,
        "input": text
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            st.error(f"Embedding API error: {response.status_code}")
            return None
        
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def batch_get_embeddings(texts: List[str], openai_api_key: str, model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Generate embeddings for multiple texts in batch"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    # OpenAI has a limit on batch size, process in chunks of 20
    batch_size = 20
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        payload = {
            "model": model,
            "input": batch
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                st.error(f"Batch embedding API error: {response.status_code}")
                return None
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)
            
            # Add small delay to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Error generating batch embeddings: {str(e)}")
            return None
    
    return all_embeddings

# Chunking functions
def create_smart_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split guideline text into overlapping chunks while preserving context.
    - Never split in the middle of sentences
    - Preserve section headers with their content
    - Maintain overlap to preserve context at boundaries
    """
    chunks = []
    
    # First, try to identify sections
    section_pattern = r'\n\s*(?:#{1,6}|[A-Z][A-Z0-9\s]+:|\d+\.)\s*([^\n]+)\n'
    sections = re.split(section_pattern, text)
    
    # If no clear sections found, chunk the entire text
    if len(sections) <= 1:
        sections = [text]
    
    current_position = 0
    
    for section_idx, section_text in enumerate(sections):
        section_text = section_text.strip()
        if not section_text:
            continue
        
        # Extract section title if present
        section_title = None
        lines = section_text.split('\n', 1)
        if len(lines) > 1 and len(lines[0]) < 200:  # Likely a title
            section_title = lines[0].strip()
            section_text = lines[1] if len(lines) > 1 else ""
        
        # Chunk the section
        section_start = current_position
        i = 0
        
        while i < len(section_text):
            # Calculate chunk boundaries
            chunk_start = max(0, i - overlap if i > 0 else 0)
            chunk_end = min(len(section_text), i + chunk_size)
            
            # Adjust end to sentence boundary
            if chunk_end < len(section_text):
                # Look for sentence endings
                sentence_ends = ['.', '!', '?', '\n\n']
                best_end = chunk_end
                
                # Search backwards for sentence end
                for j in range(chunk_end, max(chunk_start + int(chunk_size * 0.8), chunk_start), -1):
                    if j < len(section_text) and section_text[j-1:j] in sentence_ends:
                        best_end = j
                        break
                
                chunk_end = best_end
            
            # Extract chunk
            chunk_text = section_text[chunk_start:chunk_end].strip()
            
            if chunk_text:
                chunk_data = {
                    'text': chunk_text,
                    'section_title': section_title,
                    'chunk_index': len(chunks),
                    'char_start': section_start + chunk_start,
                    'char_end': section_start + chunk_end,
                    'section_index': section_idx
                }
                chunks.append(chunk_data)
            
            # Move to next chunk position
            if chunk_end >= len(section_text):
                break
            i = chunk_end - overlap
        
        current_position += len(section_text)
    
    return chunks

# Guidelines directory setup
def ensure_guidelines_directories():
    """Create necessary directories if they don't exist"""
    for directory in [GUIDELINES_DIR, GUIDELINES_PDF_DIR, GUIDELINES_TEXT_DIR, GUIDELINES_TEMP_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Initialize the index file if it doesn't exist
    if not os.path.exists(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX)):
        with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'w') as f:
            json.dump([], f)

# PDF Processing Functions
def extract_text_from_pdf(pdf_path, use_ocr=False):
    """Extract text from a PDF file using multiple methods"""
    extracted_text = ""
    
    # Try pdfplumber first (good for most PDFs with text)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
            extracted_text = "\n\n".join(pages_text)
        
        # If we got text, return it
        if extracted_text.strip():
            return extracted_text
    except Exception as e:
        st.warning(f"pdfplumber extraction attempt failed: {str(e)}")
    
    # Fall back to PyPDF2 if pdfplumber fails
    try:
        text_list = []
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text() or ""
                text_list.append(text)
            extracted_text = "\n\n".join(text_list)
        
        # If we got text, return it
        if extracted_text.strip():
            return extracted_text
    except Exception as e:
        st.warning(f"PyPDF2 extraction attempt failed: {str(e)}")
    
    # If all else fails and OCR is enabled, try OCR
    if use_ocr and ocr_support:
        try:
            text_list = []
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                # Convert image to text using OCR
                text = pytesseract.image_to_string(image)
                text_list.append(text)
            extracted_text = "\n\n".join(text_list)
            
            return extracted_text
        except Exception as e:
            st.error(f"OCR extraction attempt failed: {str(e)}")
    
    return extracted_text

def process_pdf_guideline(uploaded_file, society_name, category, year, description, use_ocr=False):
    """Process uploaded PDF file, extract text, and save to guidelines directory"""
    ensure_guidelines_directories()
    
    # Generate filenames based on society name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_filename = f"{society_name.lower().replace(' ', '_')}_{timestamp}"
    pdf_filename = f"{base_filename}.pdf"
    text_filename = f"{base_filename}.txt"
    
    pdf_path = os.path.join(GUIDELINES_PDF_DIR, pdf_filename)
    text_path = os.path.join(GUIDELINES_TEXT_DIR, text_filename)
    
    # Save the PDF file
    with open(pdf_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path, use_ocr)
    
    if not extracted_text or len(extracted_text.strip()) < 50:
        # Not enough text extracted - might be a scanned PDF
        if not use_ocr and ocr_support:
            # Try again with OCR
            st.info("Initial text extraction yielded limited results. Attempting OCR...")
            extracted_text = extract_text_from_pdf(pdf_path, use_ocr=True)
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            # Still not enough text
            return {
                "success": False,
                "error": "Could not extract sufficient text from PDF. The file may be scanned or have security settings preventing text extraction."
            }
    
    # Save the extracted text
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    # Update the index with metadata
    with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'r') as f:
        guidelines = json.load(f)
    
    # Create preview text (first 500 chars)
    preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
    
    guidelines.append({
        "id": timestamp,
        "filename": text_filename,
        "pdf_filename": pdf_filename,
        "society": society_name,
        "category": category,
        "year": year or datetime.now().year,
        "description": description or "",
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_path": text_path,
        "pdf_path": pdf_path,
        "preview": preview_text,
        "source_type": "PDF"
    })
    
    with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'w') as f:
        json.dump(guidelines, f, indent=2)
    
    # Count pages and words for reporting
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            page_count = len(reader.pages)
    except:
        page_count = "Unknown"
    
    word_count = len(re.findall(r'\b\w+\b', extracted_text))
    
    return {
        "success": True,
        "text": extracted_text,
        "text_path": text_path,
        "pdf_path": pdf_path,
        "page_count": page_count,
        "word_count": word_count,
        "preview": preview_text[:100] + "..." if len(preview_text) > 100 else preview_text,
        "guideline_id": timestamp,
        "society": society_name,
        "category": category,
        "year": year
    }

def upload_guideline_to_pinecone(guideline_data: Dict[str, Any], openai_api_key: str) -> bool:
    """Upload guideline chunks to Pinecone with embeddings"""
    pinecone_index = init_pinecone()
    if not pinecone_index:
        st.error("Pinecone not initialized")
        return False
    
    try:
        # Extract guideline text
        text = guideline_data.get('text', '')
        if not text and 'text_path' in guideline_data:
            with open(guideline_data['text_path'], 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Create chunks
        chunks = create_smart_chunks(text)
        
        # Extract medical entities for metadata
        medical_entities = extract_medical_entities_for_metadata(text)
        
        # Prepare texts for embedding
        chunk_texts = []
        for chunk in chunks:
            # Add section title to chunk text for better context
            if chunk['section_title']:
                chunk_text = f"Section: {chunk['section_title']}\n\n{chunk['text']}"
            else:
                chunk_text = chunk['text']
            chunk_texts.append(chunk_text)
        
        # Generate embeddings in batch
        with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
            embeddings = batch_get_embeddings(chunk_texts, openai_api_key)
            
        if not embeddings:
            st.error("Failed to generate embeddings")
            return False
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique ID
            chunk_id = f"{guideline_data['guideline_id']}_chunk_{i}"
            
            # Determine if this is a key recommendation
            is_key_recommendation = any(keyword in chunk['text'].lower() for keyword in 
                                      ['recommend', 'should', 'must', 'guideline', 'evidence level', 'grade'])
            
            # Extract evidence level if present
            evidence_level = extract_evidence_level(chunk['text'])
            
            # Create metadata
            metadata = {
                "chunk_id": chunk_id,
                "guideline_id": guideline_data['guideline_id'],
                "society": guideline_data['society'],
                "category": guideline_data['category'],
                "year": int(guideline_data['year']),
                "document_title": f"{guideline_data['society']} Guidelines",
                "section_title": chunk['section_title'] or "",
                "chunk_index": chunk['chunk_index'],
                "total_chunks": len(chunks),
                "chunk_text": chunk['text'][:1000],  # Store first 1000 chars for retrieval
                "full_text": chunk['text'],  # Store full text
                "quality_score": score_guideline_quality(guideline_data),
                "is_key_recommendation": is_key_recommendation,
                "evidence_level": evidence_level,
                "conditions_mentioned": medical_entities.get('conditions', [])[:5],  # Limit to 5
                "medications_mentioned": medical_entities.get('medications', [])[:5],  # Limit to 5
                "char_start": chunk['char_start'],
                "char_end": chunk['char_end']
            }
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upload to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            pinecone_index.upsert(vectors=batch)
            
        st.success(f"Successfully uploaded {len(vectors)} chunks to Pinecone")
        return True
        
    except Exception as e:
        st.error(f"Error uploading to Pinecone: {str(e)}")
        return False

def upload_guideline(uploaded_file, society_name, category, year=None, description=None):
    """Process and save an uploaded guideline file (TXT or PDF)"""
    ensure_guidelines_directories()
    
    # Generate a filename based on society name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Check file type and process accordingly
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    openai_api_key = st.session_state.get('openai_api_key')
    if not openai_api_key:
        return {
            "success": False,
            "error": "OpenAI API key required for embedding generation"
        }
    
    if file_extension == '.pdf':
        # Handle PDF file
        if not pdf_support:
            return {
                "success": False,
                "error": "PDF support is not enabled. Please install PyPDF2 and pdfplumber: pip install PyPDF2 pdfplumber"
            }
        
        result = process_pdf_guideline(uploaded_file, society_name, category, year, description)
        
        if result.get("success") and pinecone_support:
            # Upload to Pinecone
            upload_success = upload_guideline_to_pinecone(result, openai_api_key)
            result["pinecone_uploaded"] = upload_success
        
        return result
    
    elif file_extension == '.txt':
        # Handle TXT file (original functionality)
        filename = f"{society_name.lower().replace(' ', '_')}_{timestamp}.txt"
        file_path = os.path.join(GUIDELINES_TEXT_DIR, filename)
        
        # Read content for preview
        content = uploaded_file.getvalue().decode('utf-8')
        preview_text = content[:500] + "..." if len(content) > 500 else content
        
        # Save the file content
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Update the index with metadata
        with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'r') as f:
            guidelines = json.load(f)
        
        guidelines.append({
            "id": timestamp,
            "filename": filename,
            "society": society_name,
            "category": category,
            "year": year or datetime.now().year,
            "description": description or "",
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path,
            "preview": preview_text,
            "source_type": "TXT"
        })
        
        with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'w') as f:
            json.dump(guidelines, f, indent=2)
        
        word_count = len(re.findall(r'\b\w+\b', content))
        
        result = {
            "success": True,
            "text": content,
            "text_path": file_path,
            "word_count": word_count,
            "preview": preview_text[:100] + "..." if len(preview_text) > 100 else preview_text,
            "guideline_id": timestamp,
            "society": society_name,
            "category": category,
            "year": year or datetime.now().year
        }
        
        if pinecone_support:
            # Upload to Pinecone
            upload_success = upload_guideline_to_pinecone(result, openai_api_key)
            result["pinecone_uploaded"] = upload_success
        
        return result
    
    else:
        # Unsupported file type
        return {
            "success": False,
            "error": f"Unsupported file type: {file_extension}. Please upload a .txt or .pdf file."
        }

def extract_medical_entities_for_metadata(text: str) -> Dict[str, List[str]]:
    """Extract medical entities for Pinecone metadata"""
    entities = {
        'conditions': [],
        'medications': [],
        'specialized_terms': []  # ADD THIS NEW CATEGORY
    }
    
    # ADD THIS NEW SECTION FOR SPECIALIZED CONDITIONS
    # These are rare or specific medical terms that might not be caught otherwise
    specialized_condition_patterns = [
        # Lipid disorders
        r'\b(chylomicronemia|chylomicrons?|milky\s+(?:blood|serum)|lactescent)\b',
        r'\b(hypertriglyceridemia|severe\s+hypertriglyceridemia)\b',
        r'\b(familial\s+chylomicronemia\s+syndrome|FCS)\b',
        r'\b(lipoprotein\s+lipase\s+deficiency|LPL\s+deficiency)\b',
        r'\b(apolipoprotein\s+C-?II\s+deficiency|apo\s*C-?II)\b',
        r'\b(triglyceride\s+levels?\s+(?:above|greater|>)\s*\d+)\b',
        
        # Other specialized conditions (add more as needed)
        r'\b(fabry\s+disease|gaucher\s+disease|pompe\s+disease)\b',
        r'\b(hereditary\s+angioedema|HAE)\b',
        r'\b(primary\s+immunodeficiency|PIDD)\b',
    ]
    
    text_lower = text.lower()
    
    # Extract specialized conditions FIRST (before general patterns)
    for pattern in specialized_condition_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = ' '.join(match).strip()
            entities['specialized_terms'].append(match)
            entities['conditions'].append(match)  # Also add to conditions
    
    # KEEP YOUR EXISTING CONDITION PATTERNS (but they won't override specialized ones)
    condition_patterns = [
        r'\b(hypertension|hypotension|diabetes|obesity|cancer|stroke|arthritis|osteoporosis)\b',
        r'\b(depression|anxiety|insomnia|migraine|asthma|allergy|inflammation)\b',
        r'\b(heart disease|cardiovascular disease|coronary artery disease)\b',
        r'\b(kidney disease|liver disease|lung disease)\b'
    ]
    
    # KEEP YOUR EXISTING MEDICATION PATTERNS
    medication_patterns = [
        r'\b(aspirin|ibuprofen|acetaminophen|statin|metformin|insulin)\b',
        r'\b(lisinopril|amlodipine|losartan|metoprolol|atorvastatin)\b',
        r'\b(antibiotic|antiviral|antifungal|antimicrobial)\b',
        r'\b(beta[\s-]?blocker|ace[\s-]?inhibitor|calcium[\s-]?channel[\s-]?blocker)\b'
    ]
    
    # Continue with your existing extraction logic...
    # Extract conditions
    for pattern in condition_patterns:
        matches = re.findall(pattern, text_lower)
        entities['conditions'].extend(matches)
    
    # Extract medications
    for pattern in medication_patterns:
        matches = re.findall(pattern, text_lower)
        entities['medications'].extend(matches)
    
    # Deduplicate (KEEP THIS PART)
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def extract_evidence_level(text: str) -> str:
    """Extract evidence level from guideline text"""
    evidence_patterns = [
        r'evidence level[:\s]+([A-E])',
        r'grade[:\s]+([A-E])',
        r'level[:\s]+([A-E])',
        r'class[:\s]+([I-V]+)',
        r'recommendation[:\s]+([A-E])'
    ]
    
    text_lower = text.lower()
    
    for pattern in evidence_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return ""

def retrieve_relevant_chunks(claim: str, openai_api_key: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval with multiple strategies and query expansion
    """
    pinecone_index = init_pinecone()
    if not pinecone_index:
        return []
    
    all_results = []
    seen_ids = set()
    
    # Initialize analyzer for claim decomposition
    claim_analyzer = MedicalClaimAnalyzer()
    
    # STRATEGY 1: Original direct search (keep your existing code)
    with st.spinner("Searching guidelines with direct semantic search..."):
        claim_embedding = get_embedding(claim, openai_api_key)
        if claim_embedding:
            results = pinecone_index.query(
                vector=claim_embedding,
                top_k=k,
                include_metadata=True,
                filter={
                    "year": {"$gte": datetime.now().year - 5}  # Recent guidelines only
                }
            )
            
            for match in results.matches:
                if match.id not in seen_ids:
                    all_results.append({
                        'id': match.id,
                        'score': match.score,
                        'metadata': match.metadata,
                        'strategy': 'direct'
                    })
                    seen_ids.add(match.id)
    
    # NEW STRATEGY 2: Query expansion search
    with st.spinner("Trying medical synonym variations..."):
        expanded_queries = expand_medical_query(claim)
        
        # Skip the first one (it's the original claim we already searched)
        for i, expanded_query in enumerate(expanded_queries[1:], 1):
            if i > 3:  # Limit to 3 expansions to avoid too many API calls
                break
                
            exp_embedding = get_embedding(expanded_query, openai_api_key)
            if exp_embedding:
                results = pinecone_index.query(
                    vector=exp_embedding,
                    top_k=k // 2,  # Fewer results per expansion
                    include_metadata=True
                )
                
                for match in results.matches:
                    if match.id not in seen_ids and match.score > 0.7:  # Quality threshold
                        all_results.append({
                            'id': match.id,
                            'score': match.score * 0.95,  # Slightly lower weight for expansions
                            'metadata': match.metadata,
                            'strategy': f'expansion_{i}'
                        })
                        seen_ids.add(match.id)
    
    # NEW STRATEGY 3: Extract and search for specialized medical terms
    medical_entities = extract_medical_entities_for_metadata(claim)
    specialized_terms = medical_entities.get('specialized_terms', [])
    
    if specialized_terms:
        with st.spinner(f"Searching for specialized terms: {', '.join(specialized_terms[:3])}..."):
            for term in specialized_terms[:3]:  # Top 3 specialized terms
                term_embedding = get_embedding(term, openai_api_key)
                if term_embedding:
                    results = pinecone_index.query(
                        vector=term_embedding,
                        top_k=k // 3,
                        include_metadata=True
                    )
                    
                    for match in results.matches:
                        if match.id not in seen_ids and match.score > 0.65:
                            all_results.append({
                                'id': match.id,
                                'score': match.score * 0.9,
                                'metadata': match.metadata,
                                'strategy': f'specialized_term_{term}'
                            })
                            seen_ids.add(match.id)
    
    # KEEP YOUR EXISTING DECOMPOSITION STRATEGY
    claim_components = claim_analyzer.decompose_complex_claim(claim)
    if len(claim_components) > 1:
        with st.spinner(f"Searching for {len(claim_components)} claim components..."):
            for i, component in enumerate(claim_components[:3]):  # Top 3 components
                comp_embedding = get_embedding(component, openai_api_key)
                if comp_embedding:
                    results = pinecone_index.query(
                        vector=comp_embedding,
                        top_k=k // 3,
                        include_metadata=True
                    )
                    
                    for match in results.matches:
                        if match.id not in seen_ids:
                            all_results.append({
                                'id': match.id,
                                'score': match.score * 0.9,  # Slightly lower weight
                                'metadata': match.metadata,
                                'strategy': f'component_{i}'
                            })
                            seen_ids.add(match.id)
    
    # Sort by score and return top results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Log what strategies found results (helpful for debugging)
    strategies_used = set(r['strategy'] for r in all_results)
    if strategies_used:
        st.info(f"Found results using strategies: {', '.join(strategies_used)}")
    
    return all_results[:k]

def expand_medical_query(claim_text):
    """
    Expand a health claim with medical synonyms to improve guideline matching
    This helps find guidelines even when they use different terminology
    """
    # Dictionary of common terms and their medical equivalents
    medical_synonyms = {
        # Lipid-related terms
        'high triglycerides': ['hypertriglyceridemia', 'elevated triglycerides', 'raised triglycerides'],
        'triglycerides': ['triglyceride', 'TG', 'triacylglycerol'],
        'chylomicronemia': ['severe hypertriglyceridemia', 'milky blood', 'lactescent serum', 
                            'familial chylomicronemia syndrome', 'FCS', 'type 1 hyperlipoproteinemia',
                            'fredrickson type 1', 'triglycerides above 1000', 'triglycerides > 1000'],
        'milky blood': ['chylomicronemia', 'lactescent serum', 'lipemic serum'],
        
        # Cardiovascular terms
        'heart disease': ['cardiovascular disease', 'CVD', 'coronary artery disease', 'CAD', 'cardiac disease'],
        'heart attack': ['myocardial infarction', 'MI', 'acute coronary syndrome', 'ACS'],
        'high blood pressure': ['hypertension', 'HTN', 'elevated blood pressure'],
        'stroke': ['cerebrovascular accident', 'CVA', 'brain attack'],
        
        # Diabetes terms
        'diabetes': ['diabetes mellitus', 'DM', 'type 2 diabetes', 'T2DM', 'type 1 diabetes', 'T1DM'],
        'blood sugar': ['blood glucose', 'glycemia', 'glucose levels'],
        
        # Cholesterol terms
        'high cholesterol': ['hypercholesterolemia', 'dyslipidemia', 'hyperlipidemia'],
        'bad cholesterol': ['LDL', 'low-density lipoprotein', 'LDL cholesterol', 'LDL-C'],
        'good cholesterol': ['HDL', 'high-density lipoprotein', 'HDL cholesterol', 'HDL-C'],
        
        # General terms
        'fat': ['lipid', 'adipose', 'fatty'],
        'blood fat': ['blood lipid', 'serum lipid', 'plasma lipid'],
        'treatment': ['therapy', 'management', 'intervention'],
        'guidelines': ['recommendations', 'consensus', 'position statement', 'practice guidelines'],
    }
    
    # Start with the original claim
    expanded_queries = [claim_text]
    claim_lower = claim_text.lower()
    
    # For each term in our synonym dictionary
    for term, synonyms in medical_synonyms.items():
        if term in claim_lower:
            # Create variations with each synonym
            for synonym in synonyms:
                # Replace the term with its synonym
                expanded_query = claim_lower.replace(term, synonym)
                # Capitalize first letter for readability
                expanded_query = expanded_query[0].upper() + expanded_query[1:] if expanded_query else expanded_query
                expanded_queries.append(expanded_query)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for query in expanded_queries:
        if query not in seen:
            seen.add(query)
            unique_queries.append(query)
    
    # Return top 5 variations to avoid too many searches
    return unique_queries[:5]

def get_relevant_guidelines(claim_text, categories=None, max_guidelines=5):
    """Find guidelines relevant to a claim using Pinecone semantic search"""
    openai_api_key = st.session_state.get('openai_api_key')
    if not openai_api_key:
        st.warning("OpenAI API key required for semantic search")
        return []
    
    if not pinecone_support:
        # Fallback to file-based search
        return get_relevant_guidelines_file_based(claim_text, categories, max_guidelines)
    
    # Retrieve relevant chunks from Pinecone
    chunks = retrieve_relevant_chunks(claim_text, openai_api_key, k=max_guidelines * 3)
    
    if not chunks:
        return []
    
    # Group chunks by guideline document
    guidelines_map = {}
    
    for chunk in chunks:
        guideline_id = chunk['metadata'].get('guideline_id')
        if not guideline_id:
            continue
        
        if guideline_id not in guidelines_map:
            guidelines_map[guideline_id] = {
                'id': guideline_id,
                'society': chunk['metadata'].get('society', ''),
                'category': chunk['metadata'].get('category', ''),
                'year': chunk['metadata'].get('year', datetime.now().year),
                'quality_score': chunk['metadata'].get('quality_score', 50),
                'chunks': [],
                'max_relevance_score': 0,
                'content': ""
            }
        
        # Add chunk to guideline
        guidelines_map[guideline_id]['chunks'].append(chunk)
        guidelines_map[guideline_id]['max_relevance_score'] = max(
            guidelines_map[guideline_id]['max_relevance_score'],
            chunk['score']
        )
    
    # Convert to list and sort by relevance
    relevant_guidelines = []
    
    for guideline_id, guideline_data in guidelines_map.items():
        # Sort chunks by index to maintain order
        guideline_data['chunks'].sort(key=lambda x: x['metadata'].get('chunk_index', 0))
        
        # Combine chunk texts
        combined_text = "\n\n".join([
            chunk['metadata'].get('full_text', chunk['metadata'].get('chunk_text', ''))
            for chunk in guideline_data['chunks']
        ])
        
        guideline = {
            'id': guideline_id,
            'society': guideline_data['society'],
            'category': guideline_data['category'],
            'year': guideline_data['year'],
            'quality_score': guideline_data['quality_score'],
            'relevance_score': guideline_data['max_relevance_score'] * 100,  # Convert to percentage
            'content': combined_text[:2000] if len(combined_text) > 2000 else combined_text,
            'content_preview': combined_text[:200] + "..." if len(combined_text) > 200 else combined_text,
            'source_type': 'Pinecone',
            'num_relevant_chunks': len(guideline_data['chunks'])
        }
        
        relevant_guidelines.append(guideline)
    
    # Sort by combined relevance and quality score
    relevant_guidelines.sort(
        key=lambda x: (x['relevance_score'] * 0.7 + x['quality_score'] * 0.3),
        reverse=True
    )
    
    return relevant_guidelines[:max_guidelines]

def get_relevant_guidelines_file_based(claim_text, categories=None, max_guidelines=5):
    """Original file-based guideline search as fallback"""
    all_guidelines = get_all_guidelines()
    if not all_guidelines:
        return []
        
    relevant_guidelines = []
    
    # Filter by categories if provided
    if categories:
        all_guidelines = [g for g in all_guidelines if g["category"] in categories]
    
    # Extract key medical terms from claim
    medical_terms = extract_medical_terms(claim_text)
    
    # Process each guideline
    for guideline in all_guidelines:
        try:
            file_path = guideline.get("file_path")
            if not file_path or not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
                # Count keyword matches for key medical terms (weighted higher)
                medical_term_matches = sum(3 for term in medical_terms if term.lower() in content)
                
                # Count general keyword matches
                keywords = set(re.findall(r'\b\w+\b', claim_text.lower()))
                keywords = {k for k in keywords if len(k) > 3 and k not in ['this', 'that', 'with', 'from', 'have', 'what', 'when', 'where', 'which', 'your']}
                keyword_matches = sum(1 for keyword in keywords if keyword in content)
                
                # Calculate relevance score
                relevance_score = medical_term_matches + keyword_matches
                
                if relevance_score > 0:
                    guideline["relevance_score"] = relevance_score
                    guideline["content"] = content[:2000] if len(content) > 2000 else content
                    guideline["quality_score"] = score_guideline_quality(guideline)
                    relevant_guidelines.append(guideline)
        except Exception as e:
            st.error(f"Error processing guideline {guideline.get('filename')}: {str(e)}")
            continue
    
    # Sort by combined relevance and quality score
    relevant_guidelines.sort(key=lambda x: (x.get("relevance_score", 0) * 0.7 + x.get("quality_score", 0) * 0.3), reverse=True)
    
    return relevant_guidelines[:max_guidelines]

def install_pdf_support():
    """Install PDF support libraries"""
    try:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2", "pdfplumber"])
        st.success("PDF support packages installed successfully! Please restart the app.")
        return True
    except Exception as e:
        st.error(f"Error installing PDF packages: {str(e)}")
        return False

def install_ocr_support():
    """Install OCR support libraries"""
    try:
        import sys
        import subprocess
        # Install pdf2image and pytesseract
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf2image", "pytesseract"])
        st.success("OCR support packages installed successfully! Please restart the app.")
        
        # Note about Tesseract OCR
        st.info("""
        **Note:** The pytesseract package requires Tesseract OCR to be installed on your system:
        
        - **Windows:** Download and install from https://github.com/UB-Mannheim/tesseract/wiki
        - **macOS:** Install with brew: `brew install tesseract`
        - **Linux:** Install with apt: `sudo apt install tesseract-ocr`
        
        After installing Tesseract, restart this application.
        """)
        
        return True
    except Exception as e:
        st.error(f"Error installing OCR packages: {str(e)}")
        return False

def install_pinecone_support():
    """Install Pinecone support libraries"""
    try:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pinecone-client"])
        st.success("Pinecone support packages installed successfully! Please restart the app.")
        return True
    except Exception as e:
        st.error(f"Error installing Pinecone packages: {str(e)}")
        return False

# Debug function to check Pinecone connection
def debug_pinecone_connection():
    """Debug function to verify Pinecone connection"""
    st.subheader("🔍 Pinecone Debug Info")
    
    # Check 1: Is Pinecone library imported?
    st.write("**1. Pinecone Support Status:**")
    st.write(f"Pinecone library available: {pinecone_support}")
    
    # Check 2: Is API key set?
    st.write("\n**2. API Key Status:**")
    pinecone_api_key = st.session_state.get('pinecone_api_key')
    st.write(f"Pinecone API key in session: {'✅ Yes' if pinecone_api_key else '❌ No'}")
    if pinecone_api_key:
        st.write(f"API key length: {len(pinecone_api_key)} characters")
        st.write(f"API key starts with: {pinecone_api_key[:8]}...")
    
    # Check 3: Try to initialize Pinecone
    st.write("\n**3. Initialization Test:**")
    if pinecone_api_key and pinecone_support:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_api_key)
            st.success("✅ Pinecone client created successfully")
            
            # Check 4: List existing indexes
            st.write("\n**4. Existing Indexes:**")
            try:
                indexes = pc.list_indexes()
                st.write(f"Found {len(indexes.names())} indexes:")
                for idx_name in indexes.names():
                    st.write(f"- {idx_name}")
                    
                # Check if our index exists
                if PINECONE_INDEX_NAME in indexes.names():
                    st.success(f"✅ Target index '{PINECONE_INDEX_NAME}' exists")
                    
                    # Get index stats
                    index = pc.Index(PINECONE_INDEX_NAME)
                    stats = index.describe_index_stats()
                    st.write(f"\n**Index Stats:**")
                    st.json(stats)
                else:
                    st.warning(f"❌ Target index '{PINECONE_INDEX_NAME}' does not exist")
                    
            except Exception as e:
                st.error(f"Error listing indexes: {str(e)}")
                st.error("This might be due to incorrect API key or region mismatch")
                
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
    else:
        st.warning("Cannot test Pinecone - missing API key or library")

def get_all_guidelines():
    """Retrieve all guidelines from the index"""
    ensure_guidelines_directories()
    
    try:
        with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading guidelines: {str(e)}")
        return []

def delete_guideline(guideline_id):
    """Delete a guideline by ID"""
    ensure_guidelines_directories()
    
    with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'r') as f:
        guidelines = json.load(f)
    
    # Find the guideline to delete
    guideline_to_delete = None
    for guideline in guidelines:
        if guideline["id"] == guideline_id:
            guideline_to_delete = guideline
            break
    
    if guideline_to_delete:
        # Remove both text and PDF files if they exist
        if "file_path" in guideline_to_delete and os.path.exists(guideline_to_delete["file_path"]):
            os.remove(guideline_to_delete["file_path"])
        
        if "pdf_path" in guideline_to_delete and os.path.exists(guideline_to_delete["pdf_path"]):
            os.remove(guideline_to_delete["pdf_path"])
        
        # Remove from Pinecone if supported
        if pinecone_support:
            try:
                pinecone_index = init_pinecone()
                if pinecone_index:
                    # Delete all chunks for this guideline
                    pinecone_index.delete(
                        filter={
                            "guideline_id": {"$eq": guideline_id}
                        }
                    )
            except Exception as e:
                st.error(f"Error removing from Pinecone: {str(e)}")
        
        # Remove from index
        guidelines = [g for g in guidelines if g["id"] != guideline_id]
        
        with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'w') as f:
            json.dump(guidelines, f, indent=2)
        
        return True
    
    return False

def get_guideline_content(guideline_id):
    """Get the content of a specific guideline"""
    with open(os.path.join(GUIDELINES_DIR, GUIDELINES_INDEX), 'r') as f:
        guidelines = json.load(f)
    
    for guideline in guidelines:
        if guideline["id"] == guideline_id:
            try:
                with open(guideline["file_path"], 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading guideline: {str(e)}"
    
    return "Guideline not found"

def create_pdf_preview_html(pdf_path):
    """Create an HTML preview for a PDF file"""
    if not os.path.exists(pdf_path):
        return "<p>PDF file not found</p>"
    
    # Create a base64 representation of the PDF
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    
    # Create an HTML embed tag
    html = f'''
    <embed 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="500px" 
        type="application/pdf">
    '''
    
    return html

def score_guideline_quality(guideline):
    """Score guideline quality based on various factors"""
    score = 50  # Base score
    
    # Recency (max 30 points)
    current_year = datetime.now().year
    guideline_year = int(guideline.get('year', current_year - 10))
    years_old = current_year - guideline_year
    
    if years_old <= 2:
        score += 30
    elif years_old <= 5:
        score += 20
    elif years_old <= 10:
        score += 10
    
    # Society reputation (max 20 points)
    reputable_societies = [
        'WHO', 'CDC', 'FDA', 'NIH', 'AHA', 'AMA', 'AAP', 'ACOG',
        'American Heart', 'American Cancer', 'American Diabetes', 'American College',
        'European Society', 'International Association', 'World Health'
    ]
    
    society_name = guideline.get('society', '').upper()
    if any(society in society_name for society in reputable_societies):
        score += 20
    
    return score

def extract_medical_terms(text):
    """Extract likely medical terms from text for better matching"""
    potential_medical_terms = [
        'heart', 'liver', 'kidney', 'lung', 'brain', 'blood', 'pressure', 'hypertension',
        'diabetes', 'cholesterol', 'cancer', 'stroke', 'diet', 'nutrition', 'exercise',
        'obesity', 'weight', 'vitamin', 'supplement', 'hormone', 'therapy', 'treatment',
        'surgery', 'medication', 'drug', 'vaccine', 'immunity', 'antibody', 'infection',
        'virus', 'bacteria', 'inflammation', 'pain', 'chronic', 'acute', 'syndrome',
        'disease', 'disorder', 'condition', 'symptom', 'diagnosis', 'prognosis',
        'screening', 'prevention', 'detox', 'cleanse', 'natural', 'alternative',
        'holistic', 'conventional', 'evidence', 'risk', 'benefit', 'chylomicronemia',
        'triglyceride', 'triglycerides', 'pancreatitis', 'lipid', 'hypertriglyceridemia'
    ]
    
    # Find medical terms in the text
    words = re.findall(r'\b\w+\b', text.lower())
    medical_terms = [word for word in words if word in potential_medical_terms]
    
    # Also include 2-word phrases that might be medical
    phrases = re.findall(r'\b\w+\s+\w+\b', text.lower())
    medical_phrases = [phrase for phrase in phrases if 
                       any(term in phrase for term in potential_medical_terms)]
    
    return medical_terms + medical_phrases

def detect_negative_existence_claims(claim_text):
    """
    Detect claims that falsely state something doesn't exist or isn't available
    Returns penalty points if found
    """
    import re  # Make sure re is imported
    
    claim_lower = claim_text.lower()
    
    # Patterns for false "no treatment" claims
    no_treatment_patterns = [
        (r'(?:there\s+are\s+)?no\s+treatments?\s+for', 50),
        (r'(?:there\s+is\s+)?no\s+treatment\s+for', 50),
        (r'cannot\s+be\s+treated', 50),
        (r'no\s+(?:cure|therapy|intervention|option)', 40),
        (r'untreatable', 50),
        (r'nothing\s+can\s+be\s+done', 50)
    ]
    
    # Check if claim falsely states no treatment exists
    for pattern, penalty in no_treatment_patterns:
        if re.search(pattern, claim_lower):
            # Check if it's about a treatable condition
            treatable_conditions = [
                'chylomicronemia', 'hypertriglyceridemia', 'diabetes', 
                'hypertension', 'high blood pressure', 'cholesterol',
                'depression', 'anxiety', 'cancer', 'heart disease'
            ]
            
            for condition in treatable_conditions:
                if condition in claim_lower:
                    return penalty  # Return penalty for false claim
    
    return 0  # No false claims detected

def incorporate_guidelines_into_analysis(claim, system_message):
    """Add relevant guidelines to the analysis prompt with priority"""
    # Get relevant guidelines
    relevant_guidelines = get_relevant_guidelines(claim)
    
    if not relevant_guidelines:
        return system_message, None
    
    # Format guidelines for inclusion in the prompt
    guidelines_text = ""
    formatted_guidelines = []
    
    for i, guideline in enumerate(relevant_guidelines):
        try:
            guidelines_text += f"\n\nGuideline {i+1}: {guideline['society']} ({guideline['year']})\n"
            guidelines_text += f"Category: {guideline['category']}\n"
            guidelines_text += f"Quality Score: {guideline.get('quality_score', 'N/A')}/100\n"
            
            # Get content but truncate if too long
            content = guideline.get("content", "")
            if not content and os.path.exists(guideline.get("file_path", "")):
                with open(guideline["file_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"
            
            guidelines_text += f"Content: {content}\n"
            
            # Add to formatted guidelines for display in UI
            formatted_guidelines.append({
                "society": guideline['society'],
                "year": guideline['year'],
                "category": guideline['category'],
                "relevance_score": guideline.get('relevance_score', 0),
                "quality_score": guideline.get('quality_score', 0),
                "content_preview": guideline.get('content_preview', content[:200] + "..." if len(content) > 200 else content),
                "source_type": guideline.get('source_type', 'TXT')
            })
        except Exception as e:
            st.error(f"Error formatting guideline: {str(e)}")
            continue
    
    # Add a section to the system message about PRIORITIZING guidelines
    priority_message = """
    IMPORTANT: PRIORITIZE MEDICAL SOCIETY GUIDELINES WHEN AVAILABLE
    
    The following medical society guidelines represent authoritative professional consensus and 
    should be given HIGH PRIORITY when evaluating health claims. When guidelines and PubMed evidence
    conflict, defer to the guidelines unless the PubMed evidence is significantly more recent and compelling.
    
    The guidelines should be considered highly authoritative for your assessment. Explicitly mention
    when your assessment aligns with or contradicts these guidelines.
    
    SOCIETY GUIDELINES FOUND:
    """ + guidelines_text
    
    # Place the guidelines section at the BEGINNING of the system message for higher priority
    enhanced_message = priority_message + "\n\n" + system_message
    
    return enhanced_message, formatted_guidelines

# Triglyceride Expert Class
class TriglycerideExpert:
    """Domain-specific knowledge base for triglyceride-related claims"""
    
    def __init__(self):
        # Clinical reference ranges (mg/dL)
        self.reference_ranges = {
            'normal': {'min': 0, 'max': 149, 'label': 'Normal'},
            'borderline_high': {'min': 150, 'max': 199, 'label': 'Borderline High'},
            'high': {'min': 200, 'max': 499, 'label': 'High'},
            'very_high': {'min': 500, 'max': 999, 'label': 'Very High'},
            'severe': {'min': 1000, 'max': float('inf'), 'label': 'Severe Hypertriglyceridemia'}
        }
        
        # Conversion factors
        self.conversions = {
            'mg/dl_to_mmol/l': 0.0113,
            'mmol/l_to_mg/dl': 88.57
        }
        
        # Evidence-based interventions with expected effects
        self.interventions = {
            'dietary': {
                'low_carb_diet': {'reduction': '20-50%', 'evidence': 'high', 'timeframe': '4-12 weeks'},
                'mediterranean_diet': {'reduction': '10-30%', 'evidence': 'high', 'timeframe': '8-12 weeks'},
                'fish_oil': {'reduction': '20-30%', 'evidence': 'high', 'dose': '2-4g EPA+DHA daily'},
                'alcohol_cessation': {'reduction': '20-80%', 'evidence': 'high', 'timeframe': '2-4 weeks'},
                'sugar_reduction': {'reduction': '10-20%', 'evidence': 'moderate', 'timeframe': '4-8 weeks'},
                'fiber': {'reduction': '5-10%', 'evidence': 'moderate', 'dose': '25-35g daily'}
            },
            'lifestyle': {
                'weight_loss': {'reduction': '20-30%', 'evidence': 'high', 'qualifier': 'per 10% body weight'},
                'aerobic_exercise': {'reduction': '15-20%', 'evidence': 'high', 'timeframe': '8-12 weeks'},
                'resistance_training': {'reduction': '5-15%', 'evidence': 'moderate', 'timeframe': '12 weeks'}
            },
            'medications': {
                'fibrates': {'reduction': '30-50%', 'evidence': 'high', 'examples': ['fenofibrate', 'gemfibrozil']},
                'omega3_prescription': {'reduction': '20-50%', 'evidence': 'high', 'examples': ['icosapent ethyl', 'omega-3-acid ethyl esters']},
                'statins': {'reduction': '10-30%', 'evidence': 'high', 'note': 'primarily for LDL'},
                'niacin': {'reduction': '20-40%', 'evidence': 'moderate', 'note': 'limited use due to side effects'},
                'pcsk9_inhibitors': {'reduction': '15-25%', 'evidence': 'moderate', 'note': 'primarily for LDL'}
            }
        }
        
        # Conditions associated with high triglycerides
        self.associated_conditions = {
            'primary': [
                'familial chylomicronemia syndrome',
                'familial hypertriglyceridemia',
                'familial combined hyperlipidemia',
                'familial dysbetalipoproteinemia',
                'lipoprotein lipase deficiency',
                'apolipoprotein C-II deficiency'
            ],
            'secondary': [
                'diabetes mellitus',
                'metabolic syndrome',
                'obesity',
                'hypothyroidism',
                'kidney disease',
                'liver disease',
                'pregnancy',
                'medications (steroids, estrogen, retinoids, thiazides)',
                'excessive alcohol intake'
            ]
        }
        
        # Risk associations
        self.clinical_risks = {
            'pancreatitis': {
                'threshold': 500,  # mg/dL
                'risk': 'Significant risk above 500 mg/dL, high risk above 1000 mg/dL',
                'evidence': 'high'
            },
            'cardiovascular_disease': {
                'association': 'Independent risk factor when >150 mg/dL',
                'evidence': 'high'
            },
            'nafld': {
                'association': 'Strong association with non-alcoholic fatty liver disease',
                'evidence': 'high'
            }
        }
        
        # Common myths and misconceptions
        self.myths = {
            'instant_reduction': {
                'claim': 'Triglycerides can be lowered overnight',
                'fact': 'Meaningful reduction typically takes 4-12 weeks',
                'evidence': 'high'
            },
            'supplements_only': {
                'claim': 'Supplements alone can normalize severe hypertriglyceridemia',
                'fact': 'Severe cases (>500 mg/dL) usually require medication',
                'evidence': 'high'
            },
            'no_dietary_impact': {
                'claim': 'Diet doesn\'t affect triglycerides',
                'fact': 'Diet, especially carbohydrate and alcohol intake, significantly impacts levels',
                'evidence': 'high'
            }
        }
    
    def validate_triglyceride_value(self, value_str):
        """Extract and validate triglyceride values from text"""
        import re
        
        # Pattern to match triglyceride values with units
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg/dl|mg/dL)',
            r'(\d+(?:\.\d+)?)\s*(?:mmol/l|mmol/L)',
            r'triglycerides?\s*(?:of|at|:)?\s*(\d+(?:\.\d+)?)',
            r'TG\s*(?:of|at|:)?\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, value_str, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                
                # Convert to mg/dL if needed
                if 'mmol' in value_str.lower():
                    value = value * self.conversions['mmol/l_to_mg/dl']
                
                # Classify the value
                for range_name, range_data in self.reference_ranges.items():
                    if range_data['min'] <= value <= range_data['max']:
                        return {
                            'value': value,
                            'unit': 'mg/dL',
                            'classification': range_data['label'],
                            'range': range_name,
                            'clinical_action': self.get_clinical_action(value)
                        }
        
        return None
    
    def get_clinical_action(self, value_mg_dl):
        """Get recommended clinical action based on triglyceride level"""
        if value_mg_dl < 150:
            return "No specific intervention needed for triglycerides"
        elif value_mg_dl < 200:
            return "Lifestyle modifications recommended"
        elif value_mg_dl < 500:
            return "Lifestyle modifications; consider medication if CVD risk factors present"
        elif value_mg_dl < 1000:
            return "Medication indicated to prevent pancreatitis; aggressive lifestyle changes"
        else:
            return "Urgent medical management required; high risk of acute pancreatitis"
    
    def validate_intervention_claim(self, claim_text):
        """Validate claims about triglyceride interventions"""
        validations = []
        claim_lower = claim_text.lower()
        
        # Check dietary interventions
        for intervention, data in self.interventions['dietary'].items():
            if intervention.replace('_', ' ') in claim_lower:
                validations.append({
                    'intervention': intervention,
                    'expected_reduction': data['reduction'],
                    'evidence_level': data['evidence'],
                    'timeframe': data.get('timeframe', 'varies'),
                    'dose': data.get('dose', 'N/A')
                })
        
        # Check for unrealistic claims
        unrealistic_patterns = [
            (r'(\d+)%?\s*reduction\s*in\s*(\d+)\s*days?', 'timeframe'),
            (r'lower.*?(\d+)%\s*overnight', 'instant'),
            (r'eliminate.*?triglycerides', 'complete_elimination'),
            (r'cure.*?high triglycerides', 'cure_claim')
        ]
        
        for pattern, claim_type in unrealistic_patterns:
            if re.search(pattern, claim_lower):
                validations.append({
                    'issue': claim_type,
                    'validity': 'false',
                    'explanation': self.get_myth_explanation(claim_type)
                })
        
        return validations
    
    def get_myth_explanation(self, myth_type):
        """Get explanation for common myths"""
        explanations = {
            'timeframe': 'Triglyceride changes require at least 2-4 weeks to be meaningful',
            'instant': 'Triglycerides cannot be significantly lowered overnight; changes take weeks',
            'complete_elimination': 'Triglycerides are essential fats; they cannot and should not be eliminated',
            'cure_claim': 'High triglycerides is managed, not "cured"; ongoing lifestyle/treatment needed'
        }
        return explanations.get(myth_type, 'This claim appears unrealistic based on medical evidence')
    
    def generate_expert_context(self, claim_text):
        """Generate expert context for triglyceride-related claims"""
        context = []
        claim_lower = claim_text.lower()
        
        # Check if discussing specific conditions
        for condition_type, conditions in self.associated_conditions.items():
            for condition in conditions:
                if condition.lower() in claim_lower:
                    context.append(f"Note: {condition} is a {condition_type} cause of high triglycerides")
        
        # Check if discussing pancreatitis risk
        if 'pancreatitis' in claim_lower:
            context.append(f"Clinical fact: Pancreatitis risk increases significantly with triglycerides >500 mg/dL")
        
        # Check medication mentions
        for med_class, med_data in self.interventions['medications'].items():
            if any(med.lower() in claim_lower for med in med_data.get('examples', [])):
                context.append(f"Expected {med_class} effect: {med_data['reduction']} reduction")
        
        return context
    
    def check_false_claims(self, claim_text):
        """Check for false claims about established triglyceride facts"""
        claim_lower = claim_text.lower()
        false_claims = []
        
        # Check for denial of pancreatitis risk - INCLUDING CHYLOMICRONEMIA
        pancreatitis_denial_patterns = [
            # Original triglyceride patterns
            r'triglycerides?\s*(?:do(?:es)?\s*not?|don\'t|doesn\'t)\s*cause\s*pancreatitis',
            r'no\s*(?:link|connection|relationship)\s*between\s*triglycerides?\s*and\s*pancreatitis',
            r'pancreatitis\s*is\s*not?\s*(?:caused|related|linked)\s*(?:to|with)\s*triglycerides?',
            r'high\s*triglycerides?\s*(?:are|is)\s*safe',
            r'triglycerides?\s*(?:can\'t|cannot|won\'t|will\s*not?)\s*cause\s*pancreatitis',
            
            # CRITICAL: Add chylomicronemia-specific patterns
            r'chylomicronemia\s*(?:do(?:es)?\s*not?|don\'t|doesn\'t)\s*cause\s*pancreatitis',
            r'no\s*(?:link|connection|relationship)\s*between\s*chylomicronemia\s*and\s*pancreatitis',
            r'pancreatitis\s*is\s*not?\s*(?:caused|related|linked)\s*(?:to|with)\s*chylomicronemia',
            r'chylomicronemia\s*(?:are|is)\s*safe',
            r'chylomicronemia\s*(?:can\'t|cannot|won\'t|will\s*not?)\s*cause\s*pancreatitis',
            
            # Milky blood patterns
            r'milky\s*blood\s*(?:do(?:es)?\s*not?|don\'t|doesn\'t)\s*cause\s*pancreatitis',
            r'lactescent\s*serum\s*(?:do(?:es)?\s*not?|don\'t|doesn\'t)\s*cause\s*pancreatitis'
        ]
        
        for pattern in pancreatitis_denial_patterns:
            if re.search(pattern, claim_lower):
                false_claims.append({
                    'type': 'pancreatitis_denial',
                    'severity': 'high',
                    'fact': 'Severe hypertriglyceridemia/chylomicronemia (>500 mg/dL) significantly increases pancreatitis risk',
                    'penalty': 80  # Large penalty for denying established medical facts
                })
                break  # Only need to catch it once
        
        # Check for other false claims
        cardiovascular_denial_patterns = [
            r'triglycerides?\s*(?:do(?:es)?\s*not?|don\'t|doesn\'t)\s*(?:affect|impact)\s*(?:heart|cardiovascular)',
            r'no\s*(?:link|connection)\s*between\s*triglycerides?\s*and\s*(?:heart\s*disease|cvd)',
            r'chylomicronemia\s*(?:do(?:es)?\s*not?|don\'t|doesn\'t)\s*(?:affect|impact)\s*(?:heart|cardiovascular)'
        ]
        
        for pattern in cardiovascular_denial_patterns:
            if re.search(pattern, claim_lower):
                false_claims.append({
                    'type': 'cardiovascular_denial',
                    'severity': 'medium',
                    'fact': 'Triglycerides >150 mg/dL are an independent cardiovascular risk factor',
                    'penalty': 30
                })
                break
        
        return false_claims
    
    def assess_claim_plausibility(self, claim_text, extracted_values=None):
        """Assess biological plausibility of triglyceride claims"""
        plausibility_score = 100  # Start at 100%
        issues = []
        
        # Check for false claims FIRST
        false_claims = self.check_false_claims(claim_text)
        for false_claim in false_claims:
            plausibility_score -= false_claim['penalty']
            issues.append(f"Denies established fact: {false_claim['fact']}")
        
        # Check for impossible values
        if extracted_values:
            for value_data in extracted_values:
                value = value_data.get('value', 0)
                if value < 10:
                    issues.append("Impossibly low triglyceride value")
                    plausibility_score -= 50
                elif value > 10000:
                    issues.append("Impossibly high triglyceride value")
                    plausibility_score -= 50
        
        # Check for unrealistic timeframes
        timeframe_patterns = [
            (r'overnight|instantly|immediately', -40, "Unrealistic timeframe"),
            (r'in (?:1|one|2|two|3|three) days?', -30, "Too rapid timeframe"),
            (r'hours?', -50, "Impossible timeframe")
        ]
        
        claim_lower = claim_text.lower()
        for pattern, penalty, issue in timeframe_patterns:
            if re.search(pattern, claim_lower):
                plausibility_score += penalty
                issues.append(issue)
        
        # Check for unrealistic reduction percentages
        reduction_match = re.search(r'(\d+)%\s*reduction', claim_lower)
        if reduction_match:
            reduction = int(reduction_match.group(1))
            if reduction > 80:
                issues.append(f"{reduction}% reduction is unrealistic without severe intervention")
                plausibility_score -= 30
            elif reduction > 60:
                issues.append(f"{reduction}% reduction is possible but only with medication + lifestyle")
                plausibility_score -= 10
        
        return {
            'score': max(0, plausibility_score),
            'issues': issues,
            'plausible': plausibility_score > 50,
            'false_claims': false_claims  # Add this for transparency
        }

class EvidenceQualityAnalyzer:
    """
    Analyzes PubMed evidence for nuances, effect sizes, and limitations
    """
    
    def __init__(self):
        # Cautionary language patterns
        self.cautionary_patterns = [
            r'\b(caution|cautious|cautiously)\b',
            r'\b(limited evidence|insufficient evidence|weak evidence)\b',
            r'\b(further studies? (?:needed|required|indicated|warranted))\b',
            r'\b(more research (?:needed|required|is needed))\b',
            r'\b(preliminary|pilot study|small study)\b',
            r'\b(viewed with caution|interpret(?:ed)? (?:with )?caution)\b',
            r'\b(cannot be recommended|not recommended)\b',
            r'\b(inconclusive|inconsistent)\b',
            r'\b(borderline|marginal|modest) (?:effect|benefit|improvement)\b'
        ]
        
        # Limitation patterns
        self.limitation_patterns = [
            r'\b(small sample size|limited participants|n\s*=\s*\d{1,2}\b)',
            r'\b(short(?:-term)?|brief) (?:duration|study|trial|follow-up)\b',
            r'\b(single center|single-center|pilot study)\b',
            r'\b(specific population|limited to|only in)\b',
            r'\b(cannot be generalized|limited generalizability)\b',
            r'\b(observational|retrospective) (?:study|analysis)\b'
        ]
        
        # Small effect size patterns
        self.small_effect_patterns = [
            r'\b(small|modest|minor|slight|minimal) (?:effect|benefit|improvement|reduction|increase)\b',
            r'\b(?:reduction|decrease|improvement) of (?:only )?(\d+(?:\.\d+)?%)\b',
            r'\b(\d+(?:\.\d+)?)% (?:reduction|decrease|improvement)\b',
            r'\b(non-significant|not significant|ns)\b',
            r'\b(trend toward|trending toward)\b'
        ]
        
        # Large effect size patterns
        self.large_effect_patterns = [
            r'\b(significant|substantial|marked|large|dramatic) (?:effect|benefit|improvement|reduction|increase)\b',
            r'\b(?:reduction|decrease|improvement) of (\d{2,3}(?:\.\d+)?%)\b',
            r'\b(highly significant|very significant|p\s*<\s*0\.0\d+)\b'
        ]
    
    def analyze_abstract(self, abstract_text: str) -> Dict[str, Any]:
        """
        Analyze a single abstract for quality indicators
        """
        if not abstract_text:
            return self._get_empty_analysis()
        
        abstract_lower = abstract_text.lower()
        
        analysis = {
            'cautionary_language': self._detect_cautionary_language(abstract_lower),
            'study_limitations': self._detect_study_limitations(abstract_lower),
            'effect_size': self._assess_effect_size(abstract_lower),
            'confidence_level': 'medium',  # Will be calculated
            'quality_flags': []
        }
        
        # Calculate overall confidence level
        analysis['confidence_level'] = self._calculate_confidence_level(analysis)
        
        return analysis
    
    def _detect_cautionary_language(self, abstract_lower: str) -> List[str]:
        """Detect cautionary language in abstracts"""
        found_cautions = []
        
        for pattern in self.cautionary_patterns:
            matches = re.findall(pattern, abstract_lower)
            if matches:
                found_cautions.extend(matches if isinstance(matches[0], str) else [match[0] for match in matches])
        
        return list(set(found_cautions))
    
    def _detect_study_limitations(self, abstract_lower: str) -> List[str]:
        """Detect study limitations"""
        found_limitations = []
        
        for pattern in self.limitation_patterns:
            matches = re.findall(pattern, abstract_lower)
            if matches:
                found_limitations.extend(matches if isinstance(matches[0], str) else [match[0] for match in matches])
        
        return list(set(found_limitations))
    
    def _assess_effect_size(self, abstract_lower: str) -> Dict[str, Any]:
        """Assess the magnitude of effects reported"""
        effect_assessment = {
            'magnitude': 'unknown',
            'percentages': [],
            'descriptors': []
        }
        
        # Look for small effects
        small_effects = []
        for pattern in self.small_effect_patterns:
            matches = re.findall(pattern, abstract_lower)
            if matches:
                small_effects.extend(matches if isinstance(matches[0], str) else [match for match in matches if match])
        
        # Look for large effects
        large_effects = []
        for pattern in self.large_effect_patterns:
            matches = re.findall(pattern, abstract_lower)
            if matches:
                large_effects.extend(matches if isinstance(matches[0], str) else [match for match in matches if match])
        
        # Extract percentage values
        percentage_pattern = r'\b(\d+(?:\.\d+)?)%\b'
        percentages = [float(match) for match in re.findall(percentage_pattern, abstract_lower)]
        
        # Determine overall magnitude
        if large_effects or any(p > 20 for p in percentages):
            effect_assessment['magnitude'] = 'large'
        elif small_effects or any(p < 10 for p in percentages):
            effect_assessment['magnitude'] = 'small'
        elif percentages:
            effect_assessment['magnitude'] = 'moderate'
        
        effect_assessment['percentages'] = percentages
        effect_assessment['descriptors'] = small_effects + large_effects
        
        return effect_assessment
    
    def _calculate_confidence_level(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall confidence level based on analysis"""
        score = 100  # Start at 100%
        
        # Penalize for cautionary language
        caution_count = len(analysis['cautionary_language'])
        score -= caution_count * 15  # -15 points per caution
        
        # Penalize for study limitations
        limitation_count = len(analysis['study_limitations'])
        score -= limitation_count * 10  # -10 points per limitation
        
        # Penalize for small effect sizes
        if analysis['effect_size']['magnitude'] == 'small':
            score -= 20
        elif analysis['effect_size']['magnitude'] == 'moderate':
            score -= 5
        
        # Categorize confidence level
        if score >= 80:
            return 'high'
        elif score >= 60:
            return 'medium'
        elif score >= 40:
            return 'low'
        else:
            return 'very_low'
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for missing abstracts"""
        return {
            'cautionary_language': [],
            'study_limitations': [],
            'effect_size': {'magnitude': 'unknown', 'percentages': [], 'descriptors': []},
            'confidence_level': 'medium',
            'quality_flags': []
        }
    
    def analyze_multiple_abstracts(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multiple PubMed articles and provide summary
        """
        if not articles:
            return {'overall_confidence': 'low', 'analyses': [], 'summary': {}}
        
        analyses = []
        for article in articles:
            abstract = article.get('abstract', '')
            analysis = self.analyze_abstract(abstract)
            analysis['pmid'] = article.get('pmid', 'Unknown')
            analysis['title'] = article.get('title', '')[:100] + '...' if len(article.get('title', '')) > 100 else article.get('title', '')
            analyses.append(analysis)
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(analyses)
        
        return {
            'overall_confidence': summary['overall_confidence'],
            'analyses': analyses,
            'summary': summary
        }
    
    def _calculate_summary_stats(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all analyses"""
        if not analyses:
            return {'overall_confidence': 'low'}
        
        confidence_levels = [a['confidence_level'] for a in analyses]
        caution_counts = [len(a['cautionary_language']) for a in analyses]
        limitation_counts = [len(a['study_limitations']) for a in analyses]
        
        # Calculate percentages
        high_confidence_pct = (confidence_levels.count('high') / len(confidence_levels)) * 100
        has_cautions_pct = (sum(1 for c in caution_counts if c > 0) / len(caution_counts)) * 100
        has_limitations_pct = (sum(1 for l in limitation_counts if l > 0) / len(limitation_counts)) * 100
        
        # Determine overall confidence
        if high_confidence_pct >= 70:
            overall_confidence = 'high'
        elif high_confidence_pct >= 40:
            overall_confidence = 'medium'
        else:
            overall_confidence = 'low'
        
        # Adjust based on cautions and limitations
        if has_cautions_pct > 50 or has_limitations_pct > 50:
            if overall_confidence == 'high':
                overall_confidence = 'medium'
            elif overall_confidence == 'medium':
                overall_confidence = 'low'
        
        return {
            'overall_confidence': overall_confidence,
            'total_studies': len(analyses),
            'high_confidence_studies': confidence_levels.count('high'),
            'studies_with_cautions': sum(1 for c in caution_counts if c > 0),
            'studies_with_limitations': sum(1 for l in limitation_counts if l > 0),
            'high_confidence_percentage': high_confidence_pct,
            'cautions_percentage': has_cautions_pct,
            'limitations_percentage': has_limitations_pct
        }

# Enhanced Medical Claim Analyzer Class
class MedicalClaimAnalyzer:
    """Enhanced medical claim analysis with structured evaluation"""
    
    def __init__(self):
        # Common red flag phrases in health misinformation
        self.red_flag_patterns = {
        'miracle_cure': [
            r'cure.{0,10}(everything|all|any)',
            r'miracle cure', r'magic bullet', r'secret remedy',
            r'one simple trick', r'doctors hate'
        ],
        'conspiracy': [
            r'big pharma', r'they don.?t want you to know',
            r'hidden.{0,10}truth', r'suppressed.{0,10}cure',
            r'medical establishment'
        ],
        'absolute_claims': [
            r'100% effective', r'always works', r'never fails',
            r'guaranteed', r'no side effects', r'completely safe'
        ],
        'pseudoscience': [
            r'detox(?:ify)?', r'cleanse', r'alkaline', r'quantum healing',
            r'energy medicine', r'frequency therapy'
        ],
        'temporal_impossibility': [
            r'overnight', r'instantly', r'immediate(?:ly)?',
            r'in hours', r'quick fix'
        ],
        'false_medical_claims': [
            r'(?:do(?:es)?\s*not?|don\'t|doesn\'t)\s*(?:cause|lead to|result in|create|trigger)',
            r'(?:is\s*not?|isn\'t|are\s*not?|aren\'t)\s*(?:related|linked|connected|associated)',
            r'(?:has\s*no|have\s*no)\s*(?:effect|impact|relationship|connection)',
            r'myth\s*that',
            r'false(?:ly)?\s*(?:claim|believe|think)',
            r'no\s*(?:evidence|proof|studies|research)\s*(?:that|for)',
            r'debunked',
            r'contrary to (?:popular|common) belief'
        ],
        'medical_inaccuracy': [
            r'(?:cancer|diabetes|heart disease).{0,50}(?:cured|eliminated|reversed).{0,20}(?:naturally|overnight|quickly)',
            r'(?:stops|prevents|cures).{0,20}(?:all|any|every).{0,20}(?:disease|cancer|illness)',
            r'(?:starve|kill).{0,20}cancer.{0,20}cell',
            r'glucose.{0,30}(?:starve|kill).{0,20}cancer',
            r'cancer.{0,30}feed.{0,20}(?:sugar|glucose)',
            r'(?:eliminate|remove|flush).{0,20}(?:toxin|poison).{0,20}(?:from|out)'
        ]
    }
        
        # Evidence quality indicators
        self.quality_indicators = {
            'high': ['systematic review', 'meta-analysis', 'clinical guideline', 'cochrane', 'practice guideline'],
            'medium': ['randomized', 'controlled trial', 'cohort study', 'clinical trial'],
            'low': ['case report', 'opinion', 'editorial', 'anecdotal', 'observational']
        }
        
        # Initialize triglyceride expert
        self.triglyceride_expert = TriglycerideExpert()
    
    def decompose_complex_claim(self, claim_text):
        """
        Break down complex claims into individual testable components
        """
        components = []
        
        # Split by conjunctions and common claim separators
        separators = [' and ', ' also ', ' plus ', ' furthermore ', ' in addition ', ' as well as ']
        
        current_claim = claim_text
        for separator in separators:
            if separator in current_claim.lower():
                parts = re.split(separator, current_claim, flags=re.IGNORECASE)
                components.extend(parts[:-1])
                current_claim = parts[-1]
        
        components.append(current_claim)
        
        # Further decompose by identifying distinct medical claims
        decomposed = []
        for component in components:
            # Look for multiple effects/outcomes
            effects_pattern = r'(reduces?|prevents?|cures?|treats?|eliminates?|improves?|boosts?|reverses?|stops?)'
            effects = re.findall(effects_pattern + r'\s+(\w+(?:\s+\w+){0,2})', component.lower())
            
            if len(effects) > 1:
                # Multiple effects mentioned - split them
                base_text = re.split(effects_pattern, component, flags=re.IGNORECASE)[0]
                for effect in effects:
                    decomposed.append(f"{base_text} {effect[0]} {effect[1]}")
            else:
                decomposed.append(component.strip())
        
        # Clean and filter
        decomposed = [c.strip() for c in decomposed if len(c.strip()) > 10]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_components = []
        for item in decomposed:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique_components.append(item)
        
        return unique_components
    
    def detect_red_flags(self, claim):
        """Detect red flag patterns in claims"""
        red_flags_found = {}
        claim_lower = claim.lower()
        
        for flag_type, patterns in self.red_flag_patterns.items():
            for pattern in patterns:
                if re.search(pattern, claim_lower):
                    if flag_type not in red_flags_found:
                        red_flags_found[flag_type] = []
                    red_flags_found[flag_type].append(pattern)
        
        # Add specific check for triglyceride/chylomicronemia-pancreatitis denial
        if ('triglyceride' in claim_lower or 'chylomicronemia' in claim_lower) and 'pancreatitis' in claim_lower:
            denial_patterns = [
                r'not\s*cause',
                r'don\'t\s*cause',
                r'doesn\'t\s*cause',
                r'no\s*(?:link|connection|relationship)',
                r'is\s*not\s*(?:caused|related|linked)'
            ]
            
            for pattern in denial_patterns:
                if re.search(pattern, claim_lower):
                    if 'medical_falsehood' not in red_flags_found:
                        red_flags_found['medical_falsehood'] = []
                    red_flags_found['medical_falsehood'].append('Denies established medical fact')
                    break
        
        
        return red_flags_found
    
    def detect_false_medical_claims(self, claim_text):
        """Detect claims that deny established medical facts"""
        claim_lower = claim_text.lower()
        false_claim_penalty = 0
        
        # Common false medical claims
        false_patterns = {
            'vaccines.*(?:do(?:es)?\s*not?|don\'t).*(?:work|prevent)': 30,
            'vaccines.*cause.*autism': 50,
            'smoking.*(?:do(?:es)?\s*not?|doesn\'t).*(?:cause|lead to).*cancer': 50,
            'diabetes.*(?:is\s*not?|isn\'t).*(?:related|linked).*(?:to|with).*(?:diet|sugar)': 30,
            'high blood pressure.*(?:is\s*not?|isn\'t).*dangerous': 40,
            'obesity.*(?:is\s*not?|isn\'t).*(?:unhealthy|a\s*risk)': 40
        }
        
        for pattern, penalty in false_patterns.items():
            if re.search(pattern, claim_lower):
                false_claim_penalty += penalty
        
        return false_claim_penalty
    
    def detect_negative_existence_claims(claim_text):
        """
        Detect claims that falsely state something doesn't exist or isn't available
        Returns penalty points if found
        """
        claim_lower = claim_text.lower()
        
        # Patterns for false "no treatment" claims
        no_treatment_patterns = [
            (r'(?:there\s+are\s+)?no\s+treatments?\s+for', 50),
            (r'(?:there\s+is\s+)?no\s+treatment\s+for', 50),
            (r'cannot\s+be\s+treated', 50),
            (r'no\s+(?:cure|therapy|intervention|option)', 40),
            (r'untreatable', 50),
            (r'nothing\s+can\s+be\s+done', 50)
        ]
        
        # Check if claim falsely states no treatment exists
        for pattern, penalty in no_treatment_patterns:
            if re.search(pattern, claim_lower):
                # Check if it's about a treatable condition
                treatable_conditions = [
                    'chylomicronemia', 'hypertriglyceridemia', 'diabetes', 
                    'hypertension', 'high blood pressure', 'cholesterol',
                    'depression', 'anxiety', 'cancer', 'heart disease'
                ]
                
                for condition in treatable_conditions:
                    if condition in claim_lower:
                        return penalty  # Return penalty for false claim
        
        return 0  # No false claims detected

    def check_claim_plausibility(self, claim, guidelines, evidence):
        """
        Perform structured plausibility checks on medical claims
        """
        checks = {
            'mechanism_explained': False,
            'dosage_specified': False,
            'timeline_realistic': True,  # Default to true unless red flag found
            'side_effects_mentioned': False,
            'evidence_quality': 'none',
            'guideline_alignment': 'not_applicable'  # Changed from 'none' to be more specific
        }
        
        claim_lower = claim.lower()
        
        # Check if this is a triglyceride-related claim (INCLUDING CHYLOMICRONEMIA)
        triglyceride_indicators = ['triglyceride', 'tg', 'chylomicronemia', 'milky blood', 'lactescent', 'hypertriglyceridemia']
        is_triglyceride_claim = any(indicator in claim_lower for indicator in triglyceride_indicators)
        
        if is_triglyceride_claim:
            # Use triglyceride expert for specialized assessment
            tg_plausibility = self.triglyceride_expert.assess_claim_plausibility(claim)
            checks['timeline_realistic'] = tg_plausibility['plausible']
            
            # Check for intervention validation
            validations = self.triglyceride_expert.validate_intervention_claim(claim)
            if validations:
                checks['mechanism_explained'] = any(v.get('intervention') for v in validations)
                checks['dosage_specified'] = any(v.get('dose') != 'N/A' for v in validations)
        
        # Check for mechanism of action
        mechanism_keywords = ['works by', 'mechanism', 'because', 'through', 'via', 'by increasing', 'by reducing']
        checks['mechanism_explained'] = any(kw in claim_lower for kw in mechanism_keywords)
        
        # Check for dosage/amount specification
        dosage_patterns = [
            r'\d+\s*(?:mg|g|ml|iu|mcg|units?)',
            r'\d+\s*(?:times|x)\s*(?:daily|per day|a day)',
            r'\d+\s*(?:tablespoons?|teaspoons?|cups?)',
            r'\d+\s*(?:capsules?|tablets?|pills?)'
        ]
        checks['dosage_specified'] = any(re.search(p, claim_lower) for p in dosage_patterns)
        
        # Check for unrealistic timelines
        unrealistic_times = ['overnight', 'instantly', 'immediately', 'in hours', 'within days', 'quick fix']
        realistic_times = ['weeks', 'months', 'gradually', 'over time', 'long-term', 'with continued use']
        
        if any(ut in claim_lower for ut in unrealistic_times):
            checks['timeline_realistic'] = False
        elif any(rt in claim_lower for rt in realistic_times):
            checks['timeline_realistic'] = True
        
        # Check for side effects acknowledgment
        safety_keywords = ['side effect', 'risk', 'caution', 'may cause', 'in some cases', 'not suitable for', 'consult']
        checks['side_effects_mentioned'] = any(kw in claim_lower for kw in safety_keywords)
        
        # Assess evidence quality - handle both list and dict types
        if evidence:
            # Handle if evidence is a list of articles
            if isinstance(evidence, list):
                quality_scores = []
                for e in evidence:
                    if isinstance(e, dict):
                        # Check publication types or abstract for quality indicators
                        pub_types = e.get('publication_types', [])
                        abstract = e.get('abstract', '')
                        combined_text = ' '.join(pub_types) + ' ' + abstract
                        quality_scores.append(self._assess_evidence_type(combined_text))
                    else:
                        quality_scores.append(self._assess_evidence_type(str(e)))
            else:
                quality_scores = [self._assess_evidence_type(str(evidence))]
                
            if any(score == 'high' for score in quality_scores):
                checks['evidence_quality'] = 'high'
            elif any(score == 'medium' for score in quality_scores):
                checks['evidence_quality'] = 'medium'
            elif quality_scores:
                checks['evidence_quality'] = 'low'
        
        # Check guideline alignment - more nuanced approach
        if guidelines:
            if isinstance(guidelines, list) and len(guidelines) > 0:
                checks['guideline_alignment'] = 'supported'
            elif isinstance(guidelines, dict) and guidelines:
                checks['guideline_alignment'] = 'supported'
            elif guidelines:  # Any other truthy value
                checks['guideline_alignment'] = 'supported'
        else:
            # No guidelines found - this is not necessarily bad
            checks['guideline_alignment'] = 'not_applicable'
        
        return checks
    
    def _assess_evidence_type(self, evidence_text):
        """Assess the quality level of evidence with better contradiction detection"""
        evidence_lower = evidence_text.lower()
        
        # Check for contradictory evidence first
        contradiction_indicators = [
            'no evidence', 'no support', 'contradicted', 'refuted', 'disproven',
            'not supported', 'insufficient evidence', 'weak evidence', 'limited evidence'
        ]
        
        if any(indicator in evidence_lower for indicator in contradiction_indicators):
            return 'contradictory'  # New category
        
        # Original quality assessment
        for quality, indicators in self.quality_indicators.items():
            if any(indicator in evidence_lower for indicator in indicators):
                return quality
        
        return 'low'
    
    def calculate_credibility_score(self, claim, analysis_results):
        """
        Calculate a more nuanced credibility score based on multiple factors
        
        Scoring philosophy:
        - Claims WITH supporting guidelines get bonus points
        - Claims WITHOUT guidelines are NOT penalized - they rely on evidence quality
        - This prevents bias against newer/niche health topics that lack formal guidelines
        """
        score = 50  # Start at neutral
        
        # Check for well-established medical practices that may not have specific PubMed articles
        # but are universally accepted in medical practice
        claim_lower = claim.lower()
        well_established_practices = [
            # Dermatology basics
            ('sunscreen', ['acne', 'skin', 'morning', 'protection']),
            ('spf', ['acne', 'skin', 'morning', 'protection']),
            ('hand hygiene', ['infection', 'prevention', 'wash']),
            ('handwashing', ['infection', 'prevention', 'bacteria']),
            # Basic nutrition
            ('fruits and vegetables', ['health', 'diet', 'nutrition']),
            ('balanced diet', ['health', 'nutrition', 'wellness']),
            # Physical activity
            ('exercise', ['cardiovascular', 'heart', 'health']),
            ('physical activity', ['health', 'cardiovascular', 'fitness']),
            # Basic medical care
            ('vaccines', ['immunization', 'protection', 'disease']),
            ('vaccination', ['immunization', 'protection', 'disease']),
            # Sleep hygiene
            ('adequate sleep', ['health', 'rest', 'recovery']),
            ('sleep', ['health', 'recovery', 'immune'])
        ]
        
        # Check if this is a well-established practice
        is_well_established = False
        for practice, supporting_terms in well_established_practices:
            if practice in claim_lower:
                # Check if any supporting terms are present
                if any(term in claim_lower for term in supporting_terms):
                    is_well_established = True
                    break
        
        # Boost evidence quality for well-established practices with poor PubMed results
        if is_well_established:
            current_evidence = analysis_results.get('evidence_quality', 'none')
            if current_evidence in ['none', 'low']:
                analysis_results = analysis_results.copy()  # Don't modify original
                analysis_results['evidence_quality'] = 'medium'  # Boost to medium level
                # Add a note that this is a well-established practice
                analysis_results['well_established_practice'] = True
        # IMMEDIATE CHECK: Is this a known false claim?
        claim_lower = claim.lower()
        known_false_patterns = [
            'chylomicronemia.*(?:do(?:es)?\s*not?|don\'t|doesn\'t).*cause.*pancreatitis',
            'triglycerides?.*(?:do(?:es)?\s*not?|don\'t|doesn\'t).*cause.*pancreatitis',
            'no\s*(?:link|connection).*between.*(?:chylomicronemia|triglycerides).*and.*pancreatitis'
        ]
        
        for pattern in known_false_patterns:
            if re.search(pattern, claim_lower):
                score = 10  # Start VERY low for known false claims
                break
        # Check if this is a triglyceride-related claim (INCLUDING CHYLOMICRONEMIA)
        triglyceride_indicators = ['triglyceride', 'tg', 'chylomicronemia', 'milky blood', 'lactescent', 'hypertriglyceridemia']
        is_triglyceride_claim = any(indicator in claim.lower() for indicator in triglyceride_indicators)
        
        if is_triglyceride_claim:
            # Use triglyceride expert assessment
            tg_plausibility = self.triglyceride_expert.assess_claim_plausibility(claim)
            score = tg_plausibility['score']
            
            # Extract triglyceride values
            values = []
            value_data = self.triglyceride_expert.validate_triglyceride_value(claim)
            if value_data:
                values.append(value_data)
            
            # Additional scoring based on expert validation
            validations = self.triglyceride_expert.validate_intervention_claim(claim)
            for validation in validations:
                if validation.get('validity') == 'false':
                    score -= 15
                elif validation.get('evidence_level') == 'high':
                    score += 10
        else:
            # Original scoring logic for non-triglyceride claims
            # Factor 1: Red flags (-40 to 0 points)
            red_flags_count = 0
            red_flags_data = analysis_results.get('red_flags', 0)
            
            if isinstance(red_flags_data, dict):
                # If it's a dictionary, count the number of flag types
                red_flags_count = len(red_flags_data)
            elif isinstance(red_flags_data, int):
                # If it's already a count, use it
                red_flags_count = red_flags_data
            elif isinstance(red_flags_data, list):
                # If it's a list, count the items
                red_flags_count = len(red_flags_data)
                
            score -= min(red_flags_count * 10, 40)  # Cap at -40
        
        # Factor 2: Evidence support (+0 to +40 points) - INCREASED from +30
        evidence_quality = analysis_results.get('evidence_quality', 'none')
        plausibility_checks = analysis_results.get('plausibility_checks', {})
        
        # Handle nested evidence quality
        if isinstance(plausibility_checks, dict) and 'evidence_quality' in plausibility_checks:
            evidence_quality = plausibility_checks.get('evidence_quality', evidence_quality)
        
        # Increased points for evidence when no guidelines exist
        quality_scores = {
            'high': 40, 
            'medium': 25, 
            'low': 10, 
            'contradictory': -20,  # Negative points for contradictory evidence
            'none': 0
        }
        evidence_points = quality_scores.get(evidence_quality, 0)
        
        # Factor 3: Guideline alignment (variable points based on availability)
        guideline_alignment = analysis_results.get('guideline_alignment', 'not_applicable')
        
        # Handle nested guideline alignment
        if isinstance(plausibility_checks, dict) and 'guideline_alignment' in plausibility_checks:
            guideline_alignment = plausibility_checks.get('guideline_alignment', guideline_alignment)
        
        # More nuanced guideline scoring
        if guideline_alignment == 'supported':
            # Guidelines exist and support the claim
            score += 20
            # Reduce evidence weight slightly when guidelines are available
            evidence_points = int(evidence_points * 0.8)
        elif guideline_alignment == 'contradicted':
            # Guidelines exist but contradict the claim (future enhancement)
            score -= 20
            # Evidence needs to be very strong to overcome contradicting guidelines
            evidence_points = int(evidence_points * 0.5)
        elif guideline_alignment == 'not_applicable':
            # No guidelines found - rely more heavily on evidence
            # Evidence points remain at full value
            pass
        
        # Add evidence points
        score += evidence_points
        
        # Factor 4: Claim characteristics
        if isinstance(plausibility_checks, dict):
            if plausibility_checks.get('mechanism_explained'):
                score += 5
            if plausibility_checks.get('dosage_specified'):
                score += 5
            if plausibility_checks.get('timeline_realistic'):
                score += 5
            else:
                score -= 10  # Penalty for unrealistic timeline
            if plausibility_checks.get('side_effects_mentioned'):
                score += 5
        
        # Factor 5: High-quality evidence bonus
        high_quality_count = analysis_results.get('high_quality_evidence', 0)
        if high_quality_count > 0:
            # More bonus when no guidelines exist
            if guideline_alignment == 'not_applicable':
                score += min(high_quality_count * 3, 15)  # Up to +15 for multiple high-quality sources
            else:
                score += min(high_quality_count * 2, 10)  # Up to +10 for multiple high-quality sources
        
        # Ensure score is within 0-100 range
        return max(0, min(100, score))
    
    def adjust_score_based_on_assessment(self, analysis_text, original_scores):
        """
        Adjust credibility scores based on the AI's actual assessment classifications.
        This prevents misleading high scores for claims assessed as problematic.
        """
        # Enhanced assessment mappings with more comprehensive patterns
        assessment_mappings = {
            # Strong positive assessments
            'supported by strong evidence': (70, 95),
            'strongly supported': (70, 95),
            'well-supported': (70, 95),
            'robust evidence': (70, 95),
            'solid evidence': (70, 90),
            
            # Moderate positive assessments  
            'partially supported with caveats': (40, 65),
            'partially supported': (45, 70),
            'some evidence supports': (35, 60),
            'limited evidence supports': (30, 55),
            'mixed evidence': (35, 65),
            'moderate evidence': (40, 60),
            
            # CRITICAL: Enhanced insufficient evidence detection
            'insufficient evidence': (10, 25),
            'lacks sufficient evidence': (10, 25),
            'evidence is insufficient': (10, 25),
            'limited evidence': (15, 30),
            'weak evidence': (5, 20),
            'unclear evidence': (10, 25),
            'needs more research': (15, 30),
            'inconclusive': (10, 25),
            'no strong evidence': (5, 20),
            'evidence does not support': (0, 15),
            'not enough evidence': (10, 25),
            'insufficient high-quality evidence': (5, 20),
            'lacks robust clinical evidence': (5, 20),
            'no conclusive evidence': (5, 20),
            'lacks strong clinical evidence': (5, 15),
            'does not robustly support': (5, 15),
            'current evidence does not support': (5, 15),
            
            # Negative assessments (should be very low)
            'contradicted by evidence': (0, 15),
            'not supported by evidence': (5, 20),
            'misleading': (0, 10),
            'false': (0, 10),
            'inaccurate': (5, 20),
            'incorrect': (5, 20),
            'this claim is inaccurate': (0, 15),
            'this claim is incorrect': (0, 15),
            'this claim is false': (0, 10),
            'dangerous misinformation': (0, 5),
            'implausible': (0, 15),
            'biologically implausible': (0, 10),
            'lacks scientific support': (5, 20),
            'no evidence': (0, 15),
            
            # Red flag terms that should lower scores
            'conspiracy': (0, 15),
            'miracle cure': (0, 15),
            'too good to be true': (0, 20),
        }
        
        adjusted_scores = {}
        analysis_lower = analysis_text.lower()
        
        # Debug: Log what we're looking for
        print(f"DEBUG: Analysis text length: {len(analysis_text)}")
        print(f"DEBUG: Original scores: {original_scores}")
        
        # If original_scores is a dictionary (individual claim scores)
        if isinstance(original_scores, dict):
            for claim_label, original_score in original_scores.items():
                # Extract the section about this specific claim
                claim_section = self._extract_claim_section_improved(analysis_text, claim_label)
                
                # Debug: Show what section we extracted
                print(f"DEBUG: Extracted section for {claim_label}: {claim_section[:200]}...")
                
                adjusted_score = self._apply_assessment_adjustment_improved(
                    claim_section.lower(), original_score, assessment_mappings, claim_label
                )
                adjusted_scores[claim_label] = adjusted_score
        
        # If original_scores is a single number
        else:
            adjusted_score = self._apply_assessment_adjustment_improved(
                analysis_lower, original_scores, assessment_mappings, "Overall"
            )
            return adjusted_score
        
        return adjusted_scores

    def _extract_claim_section_improved(self, analysis_text, claim_label):
        """Extract the section of analysis text related to a specific claim with better matching"""
        lines = analysis_text.split('\n')
        claim_section = ""
        capturing = False
        
        # Extract claim number more reliably
        import re
        claim_match = re.search(r'claim (\d+)', claim_label.lower())
        if not claim_match:
            # Fallback: return entire analysis if we can't parse claim number
            return analysis_text
            
        claim_num = claim_match.group(1)
        
        # Look for multiple patterns to start capturing
        start_patterns = [
            f'claim {claim_num}:',
            f'**claim {claim_num}:**',
            f'claim {claim_num} ',
            f'**claim {claim_num}**'
        ]
        
        # Look for patterns to stop capturing
        next_claim_num = str(int(claim_num) + 1)
        stop_patterns = [
            f'claim {next_claim_num}:',
            f'**claim {next_claim_num}:**',
            'overall analysis',
            'bottom line',
            'credibility score:',
            '**credibility score:**'
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if we should start capturing
            if not capturing:
                if any(pattern in line_lower for pattern in start_patterns):
                    capturing = True
                    claim_section += line + " "
                    continue
            
            # If we're capturing, check if we should stop
            if capturing:
                if any(pattern in line_lower for pattern in stop_patterns):
                    break
                claim_section += line + " "
        
        # If we didn't capture anything specific, try a broader search
        if not claim_section.strip():
            # Look for the claim anywhere in the text
            for i, line in enumerate(lines):
                if f"claim {claim_num}" in line.lower():
                    # Capture this line and the next 10 lines
                    for j in range(i, min(i + 10, len(lines))):
                        claim_section += lines[j] + " "
                        # Stop if we hit another claim
                        if j > i and "claim " in lines[j].lower() and ":" in lines[j]:
                            break
                    break
        
        # Final fallback: return the whole analysis
        if not claim_section.strip():
            claim_section = analysis_text
        
        return claim_section

    def _apply_assessment_adjustment_improved(self, text_section, original_score, assessment_mappings, claim_label):
        """Apply score adjustment with better debugging and more aggressive insufficient evidence handling"""
        
        # Debug: Show what text we're analyzing
        print(f"DEBUG: Analyzing text for {claim_label}: {text_section[:300]}...")
        
        # SPECIAL CASE: Check for well-established medical practices
        # If GPT says "supported by strong evidence" but original score is low (due to no PubMed articles),
        # boost it to reflect the medical consensus
        well_established_indicators = [
            'supported by strong evidence',
            'standard dermatological recommendation', 
            'universally recommended',
            'well-established',
            'standard medical practice',
            'established medical fact',
            'medical consensus',
            'widely accepted',
            'universally recognized',
            'standard practice',
            'well-known medical fact'
        ]
        
        strong_evidence_detected = any(indicator in text_section for indicator in well_established_indicators)
        
        if strong_evidence_detected and original_score < 75:
            print(f"DEBUG: Well-established practice detected with low original score ({original_score}), boosting to 85%")
            return 85  # Boost well-established practices to 85%
        
        # Find the most relevant assessment
        found_assessments = []
        
        for assessment_phrase, (min_score, max_score) in assessment_mappings.items():
            if assessment_phrase in text_section:
                found_assessments.append((assessment_phrase, min_score, max_score, len(assessment_phrase)))
                print(f"DEBUG: Found assessment phrase: '{assessment_phrase}' -> score range ({min_score}-{max_score})")
        
        # If we found assessments, use the longest/most specific match
        if found_assessments:
            # Sort by phrase length (longer = more specific)
            most_specific = max(found_assessments, key=lambda x: x[3])
            assessment_phrase, min_allowed, max_allowed, _ = most_specific
            
            print(f"DEBUG: Using most specific phrase: '{assessment_phrase}' -> range ({min_allowed}-{max_allowed})")
            
            # For insufficient evidence cases, force score to be very low
            insufficient_terms = [
                'insufficient', 'lacks', 'weak', 'no evidence', 'not supported', 
                'does not robustly support', 'current evidence does not', 'no conclusive',
                'lacks strong clinical evidence', 'lacks robust clinical evidence'
            ]
            
            if any(term in assessment_phrase for term in insufficient_terms):
                # Force very low score for insufficient evidence
                adjusted_score = min(max_allowed, 15)  # Cap at 15% maximum
                print(f"DEBUG: Insufficient evidence detected, forcing score to {adjusted_score}")
            else:
                # Normal adjustment logic
                if original_score > max_allowed:
                    adjusted_score = max_allowed
                    print(f"DEBUG: Score too high, capping at {adjusted_score}")
                elif original_score < min_allowed:
                    adjusted_score = min_allowed
                    print(f"DEBUG: Score too low, raising to {adjusted_score}")
                else:
                    adjusted_score = original_score
                    print(f"DEBUG: Score within range, keeping {adjusted_score}")
            
            return int(adjusted_score)
        
        # If no specific assessment found, look for general negative indicators
        negative_indicators = [
            'insufficient evidence', 'lacks evidence', 'weak evidence', 'no evidence',
            'not supported', 'unclear', 'uncertain', 'questionable', 'doubtful',
            'unproven', 'speculative', 'inconclusive', 'needs more research',
            'limited evidence', 'no conclusive evidence'
        ]

        # Count negative indicators
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_section)
        
        if negative_count >= 1:
            # Force low score for any negative indicators
            adjusted_score = min(original_score, 25)  # Cap at 25% for any negative indicators
            print(f"DEBUG: Found {negative_count} negative indicators, capping score at {adjusted_score}")
            return int(adjusted_score)

        print(f"DEBUG: No assessment phrases found, keeping original score {original_score}")
        return original_score

    def calculate_credibility_score_with_domain(self, claim, analysis_results, domain_validation=None):
        """Calculate credibility score with domain-specific adjustments"""
        # Start with base calculation
        score = self.calculate_credibility_score(claim, analysis_results)
        
        # CRITICAL: Check for false claims FIRST
        if domain_validation and 'false_claims' in domain_validation.get('plausibility', {}):
            false_claims = domain_validation['plausibility']['false_claims']
            if false_claims:
                # If ANY false claims detected, cap score at 20%
                score = min(score, 20)
                # Apply additional penalties
                for fc in false_claims:
                    score -= fc['penalty']
        
        # Apply domain-specific adjustments
        if domain_validation:
            # Check for false claims
            if 'false_claims' in domain_validation['plausibility'] and domain_validation['plausibility']['false_claims']:
                # Severe penalty for denying established medical facts
                total_penalty = sum(fc['penalty'] for fc in domain_validation['plausibility']['false_claims'])
                score -= total_penalty
                
            # Original plausibility adjustments
            plausibility_score = domain_validation['plausibility']['score']
            score = (score * 0.7) + (plausibility_score * 0.3)  # Weight domain expertise at 30%
            
            # Penalize unrealistic claims
            if domain_validation['plausibility']['issues']:
                score -= len(domain_validation['plausibility']['issues']) * 5
            
            # Boost for evidence-based interventions
            for intervention in domain_validation['interventions']:
                if intervention.get('evidence_level') == 'high':
                    score += 5
        
        return max(0, min(100, score))
    
    def generate_score_explanation(self, claim, score, analysis_results, domain_validation=None):
        """
        Generate detailed explanation for why a claim received a specific credibility score
        Particularly focused on scores below 85% to help users understand issues
        """
        explanation = []
        
        # Only provide detailed breakdown for scores below 85%
        if score >= 85:
            explanation.append(f"✅ **Excellent credibility ({int(score)}%)**")
            explanation.append("This claim demonstrates strong scientific support with minimal issues.")
            return explanation
        
        explanation.append(f"📊 **Credibility Score Breakdown: {int(score)}%**")
        explanation.append("")
        
        # Analyze what contributed to the lower score
        issues_found = []
        positive_factors = []
        
        # Check for red flags
        red_flags_data = analysis_results.get('red_flags', 0)
        red_flags_count = 0
        if isinstance(red_flags_data, dict):
            red_flags_count = len(red_flags_data)
            if red_flags_count > 0:
                issues_found.append(f"🚩 **{red_flags_count} red flag(s) detected** (-{red_flags_count * 10} points)")
                for flag_type, flag_data in red_flags_data.items():
                    issues_found.append(f"   • {flag_type}: {flag_data.get('description', 'Detected')}")
        elif isinstance(red_flags_data, (int, list)):
            red_flags_count = len(red_flags_data) if isinstance(red_flags_data, list) else red_flags_data
            if red_flags_count > 0:
                issues_found.append(f"🚩 **{red_flags_count} red flag(s) detected** (-{red_flags_count * 10} points)")
        
        # Check evidence quality
        evidence_quality = analysis_results.get('evidence_quality', 'none')
        if evidence_quality == 'none':
            issues_found.append("📚 **No supporting evidence found** (0 points added)")
            issues_found.append("   • Consider providing peer-reviewed sources")
        elif evidence_quality == 'low':
            issues_found.append("📚 **Low-quality evidence only** (+10 points)")
            issues_found.append("   • Higher quality studies would strengthen this claim")
        elif evidence_quality == 'medium':
            positive_factors.append("📚 **Medium-quality evidence found** (+20 points)")
        elif evidence_quality == 'high':
            positive_factors.append("📚 **High-quality evidence found** (+40 points)")
        
        # Check guideline alignment
        guideline_alignment = analysis_results.get('guideline_alignment', 'not_applicable')
        if guideline_alignment == 'contradicted':
            issues_found.append("📋 **Contradicts medical guidelines** (-20 points)")
            issues_found.append("   • This claim goes against established medical recommendations")
        elif guideline_alignment == 'supported':
            positive_factors.append("📋 **Supported by medical guidelines** (+20 points)")
        elif guideline_alignment == 'not_applicable':
            issues_found.append("📋 **No relevant medical guidelines found** (neutral)")
            issues_found.append("   • This topic may not be covered by major medical societies")
        
        # Check plausibility factors
        plausibility_checks = analysis_results.get('plausibility_checks', {})
        if isinstance(plausibility_checks, dict):
            if not plausibility_checks.get('mechanism_explained', False):
                issues_found.append("🔬 **No mechanism of action explained** (-5 points)")
                issues_found.append("   • Claims are stronger when they explain HOW something works")
            else:
                positive_factors.append("🔬 **Mechanism of action explained** (+5 points)")
            
            if not plausibility_checks.get('dosage_specified', False):
                issues_found.append("💊 **No specific dosage mentioned** (-5 points)")
                issues_found.append("   • Specific doses/amounts make claims more credible")
            else:
                positive_factors.append("💊 **Specific dosage provided** (+5 points)")
            
            if not plausibility_checks.get('timeline_realistic', True):
                issues_found.append("⏰ **Unrealistic timeline claimed** (-10 points)")
                issues_found.append("   • Avoid claims like 'overnight results' or 'instant cure'")
            else:
                positive_factors.append("⏰ **Realistic timeline mentioned** (+5 points)")
            
            if not plausibility_checks.get('side_effects_mentioned', False):
                issues_found.append("⚠️ **No side effects or risks mentioned** (-5 points)")
                issues_found.append("   • Credible health claims acknowledge potential risks")
            else:
                positive_factors.append("⚠️ **Side effects/risks acknowledged** (+5 points)")
        
        # Check for domain-specific issues (like triglyceride expertise)
        if domain_validation:
            if 'false_claims' in domain_validation.get('plausibility', {}):
                false_claims = domain_validation['plausibility']['false_claims']
                if false_claims:
                    issues_found.append("❌ **Contains medically false information** (severe penalty)")
                    for fc in false_claims:
                        issues_found.append(f"   • {fc.get('description', 'False medical claim detected')}")
            
            if domain_validation.get('plausibility', {}).get('issues'):
                issues_found.append("🔬 **Domain expert identified issues:**")
                for issue in domain_validation['plausibility']['issues']:
                    issues_found.append(f"   • {issue}")
        
        # Check for known false patterns in the claim text
        claim_lower = claim.lower()
        known_false_patterns = [
            ('chylomicronemia.*(?:do(?:es)?\s*not?|don\'t|doesn\'t).*cause.*pancreatitis', 
             'Denying the link between severe triglycerides and pancreatitis'),
            ('triglycerides?.*(?:do(?:es)?\s*not?|don\'t|doesn\'t).*cause.*pancreatitis',
             'Denying triglyceride-pancreatitis connection'),
            ('no.*(?:link|connection).*between.*(?:chylomicronemia|triglycerides).*and.*pancreatitis',
             'False denial of established medical relationship')
        ]
        
        for pattern, description in known_false_patterns:
            if re.search(pattern, claim_lower):
                issues_found.append(f"❌ **Known false medical claim** (score capped at 20%)")
                issues_found.append(f"   • {description}")
                break
        
        # Compile the explanation
        if issues_found:
            explanation.append("**⚠️ Issues that reduced the score:**")
            explanation.extend(issues_found)
            explanation.append("")
        
        if positive_factors:
            explanation.append("**✅ Positive factors:**")
            explanation.extend(positive_factors)
            explanation.append("")
        
        # Provide actionable advice
        explanation.append("**💡 How to improve credibility:**")
        
        if score < 40:
            explanation.append("🔴 **This claim has serious credibility issues:**")
            explanation.append("   • Verify the claim against reputable medical sources")
            explanation.append("   • Look for peer-reviewed research supporting the claim")
            explanation.append("   • Check if major medical organizations address this topic")
            explanation.append("   • Be cautious about claims that seem 'too good to be true'")
        elif score < 70:
            explanation.append("🟡 **This claim needs more support:**")
            explanation.append("   • Add citations to peer-reviewed studies")
            explanation.append("   • Explain the biological mechanism if known")
            explanation.append("   • Include realistic timelines and dosages")
            explanation.append("   • Acknowledge any limitations or potential risks")
        else:
            explanation.append("🟢 **This claim is fairly credible but could be strengthened:**")
            explanation.append("   • Add more high-quality research citations")
            explanation.append("   • Include specific dosages or protocols")
            explanation.append("   • Mention any contraindications or side effects")
        
        return explanation
def final_score_validation(self, individual_scores, analysis_text):
        """
        Final validation to catch any remaining score-assessment mismatches
        This is a safety net to prevent high scores for insufficient evidence claims
        """
        corrected_scores = individual_scores.copy()
        issues_found = []
        
        # Critical phrases that should NEVER have high scores
        critical_low_confidence_phrases = [
            'insufficient evidence',
            'lacks sufficient evidence',
            'evidence is insufficient',
            'weak evidence',
            'no strong evidence',
            'no conclusive evidence',
            'lacks robust clinical evidence',
            'does not robustly support',
            'current evidence does not support',
            'inconclusive',
            'needs more research',
            'unclear evidence'
        ]
        
        analysis_lower = analysis_text.lower()
        
        for claim_label, score in individual_scores.items():
            # Extract claim number
            import re
            claim_match = re.search(r'claim (\d+)', claim_label.lower())
            if not claim_match:
                continue
                
            claim_num = claim_match.group(1)
            
            # Look for this claim's section in the analysis
            claim_patterns = [
                f'claim {claim_num}:',
                f'**claim {claim_num}:**',
                f'credibility score for claim {claim_num}:'
            ]
            
            # Find the start of this claim's section
            claim_start = -1
            for pattern in claim_patterns:
                pos = analysis_lower.find(pattern)
                if pos != -1:
                    claim_start = pos
                    break
            
            if claim_start == -1:
                continue  # Skip if we can't find this claim
            
            # Find the end of this claim's section (start of next claim or end of text)
            next_claim_num = str(int(claim_num) + 1)
            next_claim_patterns = [
                f'claim {next_claim_num}:',
                f'**claim {next_claim_num}:**',
                'overall analysis',
                'bottom line'
            ]
            
            claim_end = len(analysis_lower)
            for pattern in next_claim_patterns:
                pos = analysis_lower.find(pattern, claim_start + 1)
                if pos != -1:
                    claim_end = pos
                    break
            
            # Extract just this claim's section
            claim_section = analysis_lower[claim_start:claim_end]
            
            # Check if this claim has any critical low-confidence phrases
            found_critical_phrases = []
            for phrase in critical_low_confidence_phrases:
                if phrase in claim_section:
                    found_critical_phrases.append(phrase)
            
            # If we found critical phrases AND the score is high, force it down
            if found_critical_phrases and score > 30:
                # Determine how low the score should be based on the phrases found
                max_allowed_score = 25
                
                # Extra severe phrases get even lower scores
                severe_phrases = [
                    'insufficient evidence',
                    'lacks sufficient evidence', 
                    'no strong evidence',
                    'does not robustly support',
                    'lacks robust clinical evidence'
                ]
                
                if any(phrase in found_critical_phrases for phrase in severe_phrases):
                    max_allowed_score = 15
                
                corrected_scores[claim_label] = max_allowed_score
                issues_found.append({
                    'claim': claim_label,
                    'original_score': score,
                    'corrected_score': max_allowed_score,
                    'phrases_found': found_critical_phrases,
                    'reason': f"High score ({score}%) contradicts assessment phrases: {', '.join(found_critical_phrases)}"
                })
        
        return corrected_scores, issues_found

def validate_score_assessment_alignment(individual_scores, analysis_text):
    """
    Validate that credibility scores align with evidence assessments
    Enhanced version with better pattern matching
    """
    issues_found = []
    
    # Convert to lowercase for better matching
    analysis_lower = analysis_text.lower()
    
    # Check each claim
    for claim_label, score in individual_scores.items():
        # Extract the claim number more reliably
        import re
        claim_match = re.search(r'claim (\d+)', claim_label.lower())
        if not claim_match:
            continue
            
        claim_num = claim_match.group(1)
        
        # Find the assessment text for this claim with multiple patterns
        claim_patterns = [
            rf'claim {claim_num}:.*?(?=claim \d+:|$)',
            rf'\*\*claim {claim_num}:\*\*.*?(?=\*\*claim \d+:|\*\*credibility score|$)',
            rf'claim {claim_num}[:\.].*?(?=claim \d+[:\.]|credibility score|$)'
        ]
        
        claim_section = None
        for pattern in claim_patterns:
            claim_match = re.search(pattern, analysis_lower, re.DOTALL | re.IGNORECASE)
            if claim_match:
                claim_section = claim_match.group(0)
                break
        
        if not claim_section:
            # Fallback: search for any mention of this claim
            claim_start = analysis_lower.find(f"claim {claim_num}")
            if claim_start != -1:
                claim_end = analysis_lower.find(f"claim {int(claim_num) + 1}", claim_start)
                if claim_end == -1:
                    claim_end = len(analysis_lower)
                claim_section = analysis_lower[claim_start:claim_end]
        
        if claim_section:
            # Enhanced insufficient evidence detection
            insufficient_patterns = [
                'insufficient evidence',
                'lacks sufficient evidence', 
                'evidence is insufficient',
                'weak evidence',
                'limited evidence',
                'no strong evidence',
                'not supported by evidence',
                'lacks strong clinical evidence',
                'does not robustly support',
                'current evidence does not support',
                'lacks robust clinical evidence',
                'insufficient high-quality evidence',
                'no specific guidelines',
                'no evidence',
                'inconclusive',
                'unclear evidence',
                'not enough evidence'
            ]
            
            # Check for insufficient evidence indicators
            has_insufficient_evidence = any(pattern in claim_section for pattern in insufficient_patterns)
            
            # More aggressive negative assessment detection
            negative_patterns = [
                'contradicted by evidence',
                'misleading',
                'false',
                'not supported',
                'debunked',
                'lacks evidence',
                'no clinical evidence',
                'not proven',
                'unsubstantiated'
            ]
            
            has_negative_assessment = any(pattern in claim_section for pattern in negative_patterns)
            
            # CRITICAL: Force low scores for insufficient evidence
            if has_insufficient_evidence and score > 35:
                issues_found.append({
                    'claim': claim_label,
                    'score': score,
                    'issue': 'High score despite insufficient evidence marking',
                    'suggested_score': max(45, int(score * 0.7)),  # Reduce by 30% instead of crashing to 20%
                    'detected_phrase': next((p for p in insufficient_patterns if p in claim_section), 'insufficient evidence')
                })
            
            # Force very low scores for negative assessments
            if has_negative_assessment and score > 25:
                issues_found.append({
                    'claim': claim_label,
                    'score': score,
                    'issue': 'High score despite negative assessment',
                    'suggested_score': 15,  # Very low score
                    'detected_phrase': next((p for p in negative_patterns if p in claim_section), 'negative assessment')
                })
            
            # Special check for specific phrases that should always result in low scores
            critical_phrases = [
                'lacks strong clinical evidence',
                'does not robustly support',
                'current evidence does not support'
            ]
            
            has_critical_phrase = any(phrase in claim_section for phrase in critical_phrases)
            if has_critical_phrase and score > 30:
                issues_found.append({
                    'claim': claim_label,
                    'score': score,
                    'issue': 'High score despite critical evidence limitation',
                    'suggested_score': 25,
                    'detected_phrase': next((p for p in critical_phrases if p in claim_section), 'critical limitation')
                })
    
    return issues_found

# Helper functions
def save_to_history(text, analysis, credibility_scores=None, source_type="Text", source_url=None):
    """Save analysis to history file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle multiple credibility scores
    if isinstance(credibility_scores, list):
        # Calculate average for overall score
        valid_scores = [s for s in credibility_scores if isinstance(s, (int, float))]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else "N/A"
        score_details = json.dumps(credibility_scores)
    elif isinstance(credibility_scores, dict):
        # Extract scores from dict
        valid_scores = [v for v in credibility_scores.values() if isinstance(v, (int, float))]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else "N/A"
        score_details = json.dumps(credibility_scores)
    else:
        overall_score = credibility_scores or "N/A"
        score_details = str(credibility_scores)
    
    # Create DataFrame for the new entry
    new_entry = pd.DataFrame({
        "timestamp": [timestamp],
        "query_text": [text[:100] + "..." if len(text) > 100 else text],
        "credibility_score": [overall_score],
        "score_details": [score_details],
        "full_analysis": [analysis],
        "source_type": [source_type],
        "source_url": [source_url if source_url else ""]
    })
    
    # Save to CSV (append or create)
    try:
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
            # Add score_details column if it doesn't exist
            if 'score_details' not in history_df.columns:
                history_df['score_details'] = 'N/A'
            updated_df = pd.concat([history_df, new_entry], ignore_index=True)
        else:
            updated_df = new_entry
        
        updated_df.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.error(f"Failed to save to history: {str(e)}")

def load_history():
    """Load analysis history"""
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception as e:
            st.error(f"Failed to load history: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

class PubMedSearcher:
    """
    Class for searching and retrieving open access articles from PubMed with enhanced performance
    
    Enhanced Features:
    - GPT-powered intelligent search query generation
    - Robust fallback hierarchy for query generation
    - Advanced query validation and quality checking
    - Comprehensive logging and debugging capabilities
    - Medical term extraction with pattern matching
    - Pharmaceutical drug name recognition and classification
    - Specific drug-to-class mapping for better queries
    
    Example Usage:
        # Initialize with API keys and logging
        searcher = PubMedSearcher(
            email="your@email.com", 
            api_key="your_openai_key",
            enable_logging=True
        )
        
        # Enable detailed logging for debugging
        searcher.enable_query_logging(True)
        
        # Search for articles - will use GPT-powered query generation
        articles = searcher.find_relevant_articles("Plant sterols reduce cholesterol")
        
        # Query generation hierarchy:
        # 1. GPT-powered generation (if API key available)
        # 2. Smart medical concept extraction  
        # 3. Pharmaceutical drug name extraction (HIGH PRIORITY)
        # 4. Basic intervention-outcome extraction
        # 5. Medical term pattern matching
        # 6. Simple keyword extraction
        # 7. Truncated claim text (last resort)
    """
    
    PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    PUBMED_SEARCH_URL = PUBMED_API_URL + "esearch.fcgi"
    PUBMED_FETCH_URL = PUBMED_API_URL + "efetch.fcgi"
    PUBMED_SUMMARY_URL = PUBMED_API_URL + "esummary.fcgi"
    
    MAX_RETRIES = 3
    RETRY_DELAY = 4
    SEARCH_DELAY = 0.5  # Reduced delay - API key allows more requests per second
    
    def __init__(self, email=None, api_key=None, enable_logging=False):
        """
        Initialize PubMed searcher
        
        Args:
            email: Optional email to add to PubMed requests (recommended by NCBI)
            api_key: NCBI API key for higher rate limits
            enable_logging: Whether to enable query generation logging for debugging
        """
        self.email = email
        self.api_key = api_key
        self.cache = {}  # Simple in-memory cache
        self._enable_logging = enable_logging
        
        # Medical term synonyms dictionary for query expansion
        self.medical_synonyms = {
            # Common conditions and their synonyms
            "heart attack": ["myocardial infarction", "cardiac arrest", "coronary thrombosis"],
            "stroke": ["cerebrovascular accident", "brain attack", "cerebral infarction"],
            "high blood pressure": ["hypertension", "elevated blood pressure"],
            "diabetes": ["diabetes mellitus", "type 2 diabetes", "hyperglycemia"],
            "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma"],
            "obesity": ["overweight", "adiposity", "excess weight"],
            
            # Treatments and their synonyms
            "surgery": ["operation", "surgical procedure", "resection"],
            "medication": ["drug", "pharmaceutical", "medicine", "therapy"],
            "exercise": ["physical activity", "workout", "training", "fitness"],
            "diet": ["nutrition", "food intake", "eating pattern", "dietary pattern"],
            "supplement": ["vitamin", "mineral", "dietary supplement", "nutraceutical"],
            
            # Specific nutrients/compounds and their synonyms
            "vitamin d": ["cholecalciferol", "vitamin d3", "calciferol"],
            "vitamin c": ["ascorbic acid", "ascorbate"],
            "vitamin e": ["tocopherol", "alpha-tocopherol"],
            "omega 3": ["fish oil", "epa", "dha", "n-3 fatty acids"],
            "antioxidant": ["free radical scavenger", "polyphenol"],
            "protein": ["amino acids", "peptides", "albumin"],
            
            # Common health terms and their synonyms
            "inflammation": ["inflammatory response", "swelling"],
            "immune system": ["immunity", "immune response", "immune function"],
            "detox": ["detoxification", "cleansing", "purification"],
            "weight loss": ["weight reduction", "slimming", "fat loss"],
            "digestion": ["digestive process", "gastrointestinal function"],
            "metabolism": ["metabolic rate", "energy expenditure"],
            
            # Common supplements and alternative remedies
            "probiotics": ["beneficial bacteria", "gut flora", "microbiome"],
            "turmeric": ["curcumin", "curcuma longa"],
            "garlic": ["allium sativum", "allicin"],
            "ginger": ["zingiber officinale"],
            "echinacea": ["coneflower"],
            "ginseng": ["panax"],
            
            # Health conditions
            "arthritis": ["joint inflammation", "osteoarthritis", "rheumatoid arthritis"],
            "depression": ["major depressive disorder", "clinical depression"],
            "anxiety": ["anxiety disorder", "generalized anxiety"],
            "insomnia": ["sleep disorder", "sleeplessness", "sleep difficulty"],
            "allergy": ["allergic reaction", "hypersensitivity"],
            "migraine": ["migraine headache", "vascular headache"],
            
            # Bodily systems
            "cardiovascular": ["heart", "blood vessels", "circulatory"],
            "digestive": ["gastrointestinal", "gi tract", "gut"],
            "respiratory": ["lungs", "breathing", "pulmonary"],
            "nervous system": ["neurological", "brain", "neural"],
            "endocrine": ["hormonal", "glands", "hormone"],
            "immune": ["immunity", "lymphatic", "defense"],
            
            # Body parts for better matching
            "skin": ["dermal", "epidermis", "cutaneous"],
            "liver": ["hepatic", "hepato"],
            "kidney": ["renal", "nephro"],
            "lung": ["pulmonary", "bronchial"],
            "brain": ["cerebral", "neural", "cognitive"],
            "heart": ["cardiac", "myocardial", "coronary"],
            "blood": ["hematologic", "plasma", "serum"]
        }
    
    def _extract_medical_entities(self, text):
        """
        Extract and classify medical entities from text - improved version
        """
        text_lower = text.lower()
        
        entities = {
            'conditions': [],
            'interventions': [],
            'outcomes': [],
            'anatomical': [],
            'compound_terms': []  # New category for multi-word entities
        }
        
        # First, extract compound medical terms (multi-word entities)
        compound_patterns = [
            # Fatty acids and compounds
            r'\bomega[\s-]?3(?:\s+fatty\s+acids?)?\b',
            r'\bomega[\s-]?6(?:\s+fatty\s+acids?)?\b',
            r'\bfish\s+oil\b',
            r'\bfatty\s+acids?\b',
            r'\beicosapentaenoic\s+acid\b|\bepa\b',
            r'\bdocosahexaenoic\s+acid\b|\bdha\b',
            
            # Common supplements and compounds
            r'\bvitamin\s+[a-zA-Z]\d?\b',
            r'\bvitamin\s+(?:complex|b\s*complex)\b',
            r'\bco(?:enzyme)?\s*q10\b',
            r'\balpha[\s-]?lipoic\s+acid\b',
            r'\bhyaluronic\s+acid\b',
            r'\bfolic\s+acid\b',
            r'\bamino\s+acids?\b',
            
            # Medical conditions (multi-word)
            r'\bheart\s+disease\b',
            r'\bblood\s+pressure\b',
            r'\bhigh\s+blood\s+pressure\b',
            r'\blow\s+blood\s+pressure\b',
            r'\btype\s+2\s+diabetes\b',
            r'\brheumatoid\s+arthritis\b',
            r'\bmultiple\s+sclerosis\b',
            r'\birritable\s+bowel\s+syndrome\b',
            r'\bchronic\s+fatigue\s+syndrome\b',
            r'\bcardiovascular\s+disease\b',
            r'\bcoronary\s+artery\s+disease\b',
            
            # Clinical measurements
            r'\btriglyceride(?:s)?\s+levels?\b',
            r'\btriglycerides?\b',
            r'\bcholesterol\s+levels?\b',
            r'\bldl\s+cholesterol\b',
            r'\bhdl\s+cholesterol\b',
            r'\bblood\s+sugar\b',
            r'\bblood\s+glucose\b',
            
            # Treatment types
            r'\bclinical\s+trial\b',
            r'\bsystematic\s+review\b',
            r'\bmeta[\s-]?analysis\b',
            
            # Additional compound terms
            r'\bgreen\s+tea\b',
            r'\bblack\s+tea\b',
            r'\bweight\s+loss\b',
            r'\bbody\s+mass\s+index\b|\bbmi\b',
            r'\bimmune\s+system\b',
            r'\bdigestive\s+system\b',
            r'\bcardiovascular\s+system\b',
            
            # Plant compounds and sterols
            r'\bplant\s+sterols?\b',
            r'\bplant\s+stanols?\b',
            r'\bphytosterols?\b',
            r'\bphytostanols?\b',
            r'\bsterol\s+esters?\b',
            r'\bstanol\s+esters?\b'
        ]
        
        # Extract compound terms first (before they get broken down)
        for pattern in compound_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                clean_match = match.strip()
                if clean_match and clean_match not in entities['compound_terms']:
                    entities['compound_terms'].append(clean_match)
                    # Also categorize them appropriately
                    if any(term in clean_match for term in ['vitamin', 'omega', 'acid', 'oil', 'q10', 'supplement', 'tea', 'sterol', 'stanol']):
                        entities['interventions'].append(clean_match)
                    elif any(term in clean_match for term in ['disease', 'syndrome', 'arthritis', 'diabetes', 'pressure', 'cholesterol']):
                        entities['conditions'].append(clean_match)
                    elif any(term in clean_match for term in ['triglyceride', 'cholesterol', 'sugar', 'glucose', 'levels', 'system']):
                        entities['outcomes'].append(clean_match)
        
        # Extract single-word conditions
        condition_patterns = [
            r'\b(hypertension|hypotension|diabetes|obesity|cancer|stroke|arthritis|osteoporosis)\b',
            r'\b(depression|anxiety|insomnia|migraine|asthma|allergy|inflammation)\b',
            r'\b(atherosclerosis|thrombosis|embolism|anemia|leukemia|lymphoma)\b'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in entities['conditions'] and not any(match in term for term in entities['compound_terms']):
                    entities['conditions'].append(match)
        
        # Extract specific outcomes/effects - IMPROVED to capture better context
        outcome_patterns = [
            r'\b(reduce|reduces|reduced|reduction|decrease|decreases|decreased)\s+(?:the\s+)?(?:risk\s+of\s+)?(\w+(?:\s+\w+){0,2})\b',
            r'\b(increase|increases|increased|improve|improves|improved)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})\b',
            r'\b(lower|lowers|lowered|lowering)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})\b',
            r'\b(prevent|prevents|prevention|preventing)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})\b',
            r'\b(treat|treats|treatment|treating)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})\b',
            r'\b(regulate|regulates|regulation|regulating)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})\b',
            r'\b(support|supports|supporting)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})\b',
            r'\b(block|blocks|blocking|inhibit|inhibits|inhibiting)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})\b'
        ]
        
        for pattern in outcome_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    effect = f"{match[0]} {match[1]}".strip()
                    if effect not in entities['outcomes']:
                        entities['outcomes'].append(effect)
        
        # Extract single interventions not caught by compound patterns
        intervention_patterns = [
            r'\b(aspirin|ibuprofen|acetaminophen|statin|metformin|insulin)\b',
            r'\b(exercise|diet|nutrition|lifestyle|meditation|yoga|therapy)\b',
            r'\b(surgery|chemotherapy|radiation|immunotherapy|antibiotic)\b',
            r'\b(supplement|medication|treatment|intervention)\b',
            r'\b(fiber|fibre|psyllium|niacin|magnesium)\b'
        ]
        
        for pattern in intervention_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in entities['interventions'] and not any(match in term for term in entities['compound_terms']):
                    entities['interventions'].append(match)
        
        # Clean and deduplicate
        for key in entities:
            cleaned = []
            seen = set()
            for term in entities[key]:
                term = str(term).strip()
                if term and term not in seen and len(term) > 2:
                    cleaned.append(term)
                    seen.add(term)
            entities[key] = cleaned
        
        return entities
    
    def _extract_intervention_and_outcome(self, claim_text):
        """
        Extract the main intervention/substance and outcome/effect from a health claim.
        This is simpler and more focused than the existing _extract_key_concepts.
        """
        claim_lower = claim_text.lower()
        
        # Common intervention patterns - ordered by specificity
        interventions = {
            # Specific compounds and supplements
            'omega-3 fatty acids': ['omega-3', 'omega 3', 'fish oil', 'epa', 'dha'],
            'plant sterols': ['plant sterol', 'phytosterol'],
            'plant stanols': ['plant stanol', 'phytostanol'],
            'psyllium': ['psyllium'],
            'niacin': ['niacin', 'vitamin b3', 'nicotinic acid'],
            'fiber': ['fiber', 'fibre', 'dietary fiber', 'soluble fiber'],
            'statins': ['statin', 'atorvastatin', 'rosuvastatin', 'simvastatin'],
            'vitamin d': ['vitamin d', 'cholecalciferol', 'vitamin d3'],
            'vitamin c': ['vitamin c', 'ascorbic acid'],
            'magnesium': ['magnesium'],
            'probiotics': ['probiotic', 'lactobacillus', 'bifidobacterium'],
            'turmeric': ['turmeric', 'curcumin'],
            'garlic': ['garlic', 'allicin'],
            'green tea': ['green tea', 'egcg', 'catechin'],
            'red yeast rice': ['red yeast rice'],
            'coenzyme q10': ['coenzyme q10', 'coq10', 'ubiquinone'],
            'alpha lipoic acid': ['alpha lipoic acid', 'lipoic acid'],
            'chromium': ['chromium'],
            'cinnamon': ['cinnamon'],
            
            # Specific pharmaceutical drugs - Cardiovascular
            'evacetrapib': ['evacetrapib'],
            'ezetimibe': ['ezetimibe', 'zetia'],
            'atorvastatin': ['atorvastatin', 'lipitor'],
            'rosuvastatin': ['rosuvastatin', 'crestor'],
            'simvastatin': ['simvastatin', 'zocor'],
            'pravastatin': ['pravastatin', 'pravachol'],
            'lovastatin': ['lovastatin', 'mevacor'],
            'fluvastatin': ['fluvastatin', 'lescol'],
            'pitavastatin': ['pitavastatin', 'livalo'],
            'fenofibrate': ['fenofibrate', 'tricor'],
            'gemfibrozil': ['gemfibrozil', 'lopid'],
            'cholestyramine': ['cholestyramine', 'questran'],
            'colesevelam': ['colesevelam', 'welchol'],
            'alirocumab': ['alirocumab', 'praluent'],
            'evolocumab': ['evolocumab', 'repatha'],
            'inclisiran': ['inclisiran', 'leqvio'],
            'bempedoic acid': ['bempedoic acid', 'nexletol'],
            
            # Diabetes medications
            'metformin': ['metformin', 'glucophage'],
            'glipizide': ['glipizide', 'glucotrol'],
            'glyburide': ['glyburide', 'diabeta'],
            'pioglitazone': ['pioglitazone', 'actos'],
            'rosiglitazone': ['rosiglitazone', 'avandia'],
            'sitagliptin': ['sitagliptin', 'januvia'],
            'linagliptin': ['linagliptin', 'tradjenta'],
            'empagliflozin': ['empagliflozin', 'jardiance'],
            'canagliflozin': ['canagliflozin', 'invokana'],
            'semaglutide': ['semaglutide', 'ozempic', 'wegovy'],
            'liraglutide': ['liraglutide', 'victoza', 'saxenda'],
            'dulaglutide': ['dulaglutide', 'trulicity'],
            
            # Skincare active ingredients
            'salicylic acid': ['salicylic acid', 'bha', 'beta hydroxy acid'],
            'glycolic acid': ['glycolic acid', 'aha', 'alpha hydroxy acid'],
            'lactic acid': ['lactic acid'],
            'hyaluronic acid': ['hyaluronic acid', 'sodium hyaluronate'],
            'retinol': ['retinol', 'retinoid', 'retinoic acid', 'tretinoin'],
            'niacinamide': ['niacinamide', 'nicotinamide'],
            'vitamin e': ['vitamin e', 'tocopherol'],
            'benzoyl peroxide': ['benzoyl peroxide'],
            'azelaic acid': ['azelaic acid'],
            'kojic acid': ['kojic acid'],
            'arbutin': ['arbutin'],
            'ceramides': ['ceramide', 'ceramides'],
            'peptides': ['peptide', 'peptides'],
            'tea tree oil': ['tea tree oil', 'melaleuca'],
            'zinc oxide': ['zinc oxide'],
            
            # Skincare products and treatments
            'cleanser': ['cleanser', 'facial cleanser', 'face wash'],
            'serum': ['serum', 'treatment serum'],
            'moisturizer': ['moisturizer', 'cream', 'lotion'],
            'sunscreen': ['sunscreen', 'spf', 'sun protection'],
            'toner': ['toner', 'astringent'],
            'mask': ['mask', 'face mask', 'clay mask'],
            'exfoliant': ['exfoliant', 'scrub', 'peel'],
            
            # Dietary patterns
            'mediterranean diet': ['mediterranean diet'],
            'low-carb diet': ['low-carb', 'low carb', 'ketogenic', 'keto'],
            'dash diet': ['dash diet'],
            'plant-based diet': ['plant-based', 'vegetarian', 'vegan'],
            
            # Lifestyle interventions
            'exercise': ['exercise', 'physical activity', 'aerobic', 'resistance training'],
            'weight loss': ['weight loss', 'weight reduction'],
            'meditation': ['meditation', 'mindfulness'],
            'sleep': ['sleep'],
            'stress reduction': ['stress reduction', 'stress management']
        }
        
        # Common outcomes/effects
        outcomes = {
            'triglycerides': ['triglyceride', 'triglycerides', 'tg', 'hypertriglyceridemia'],
            'ldl cholesterol': ['ldl', 'low-density lipoprotein', 'ldl cholesterol', 'ldl-c'],
            'hdl cholesterol': ['hdl', 'high-density lipoprotein', 'hdl cholesterol', 'hdl-c'],
            'total cholesterol': ['cholesterol', 'total cholesterol', 'hypercholesterolemia'],
            'blood pressure': ['blood pressure', 'hypertension', 'bp'],
            'blood sugar': ['blood sugar', 'glucose', 'glycemia', 'diabetes', 'a1c', 'hba1c'],
            'inflammation': ['inflammation', 'inflammatory', 'crp', 'c-reactive protein'],
            'cardiovascular': ['cardiovascular', 'heart disease', 'cvd', 'cardiac'],
            'lipid profile': ['lipid profile', 'lipids', 'dyslipidemia'],
            'cholesterol absorption': ['cholesterol absorption', 'absorption'],
            'weight': ['weight', 'bmi', 'body mass index', 'obesity'],
            'anxiety': ['anxiety', 'anxious'],
            'depression': ['depression', 'depressive'],
            'cognitive': ['cognitive', 'cognition', 'memory', 'brain'],
            'immune': ['immune', 'immunity', 'immune system'],
            'cancer': ['cancer', 'tumor', 'malignancy'],
            'pancreatitis': ['pancreatitis'],
            
            # Skincare outcomes and effects
            'acne': ['acne', 'acne vulgaris', 'pimples', 'breakouts', 'blemishes'],
            'skin texture': ['skin texture', 'texture', 'smoothness', 'rough skin'],
            'blackheads': ['blackheads', 'comedones', 'clogged pores'],
            'whiteheads': ['whiteheads', 'closed comedones'],
            'pores': ['pores', 'pore size', 'enlarged pores'],
            'oily skin': ['oily skin', 'excess oil', 'sebum production'],
            'dry skin': ['dry skin', 'dryness', 'dehydration'],
            'wrinkles': ['wrinkles', 'fine lines', 'aging', 'anti-aging'],
            'hyperpigmentation': ['hyperpigmentation', 'dark spots', 'melasma', 'age spots'],
            'skin tone': ['skin tone', 'uneven skin', 'discoloration'],
            'redness': ['redness', 'irritation', 'sensitivity'],
            'exfoliation': ['exfoliation', 'dead skin cells', 'cell turnover'],
            'moisturization': ['moisturization', 'hydration', 'skin barrier'],
            'sun protection': ['sun protection', 'uv protection', 'photoaging'],
            'collagen': ['collagen', 'elasticity', 'firmness'],
            'skin clarity': ['skin clarity', 'clear skin', 'radiance']
        }
        
        # Find interventions
        found_interventions = []
        for standard_name, variations in interventions.items():
            for variation in variations:
                if variation in claim_lower:
                    found_interventions.append(standard_name)
                    break
        
        # Find outcomes
        found_outcomes = []
        for standard_name, variations in outcomes.items():
            for variation in variations:
                if variation in claim_lower:
                    found_outcomes.append(standard_name)
                    break
        
        # Also look for action words that might indicate the relationship
        actions = []
        action_patterns = [
            r'\b(lower|lowers|lowering|reduce|reduces|reducing|decrease|decreases)\b',
            r'\b(raise|raises|raising|increase|increases|increasing|improve|improves)\b',
            r'\b(prevent|prevents|prevention|block|blocks|blocking|inhibit|inhibits)\b',
            r'\b(treat|treats|treatment|manage|manages|management)\b',
            r'\b(cause|causes|causing|lead to|leads to|result in|results in)\b'
        ]
        
        for pattern in action_patterns:
            if re.search(pattern, claim_lower):
                match = re.search(pattern, claim_lower)
                actions.append(match.group(1))
        
        return {
            'interventions': found_interventions,
            'outcomes': found_outcomes,
            'actions': actions,
            'has_content': bool(found_interventions or found_outcomes)
        }
    
    def _extract_medical_concepts_advanced(self, claim_text):
        """
        Advanced medical concept extraction that preserves compound terms
        and understands medical relationships
        """
        claim_lower = claim_text.lower()
        
        # Dictionary of compound medical concepts that should stay together
        compound_medical_terms = {
            # Diet-related compounds
            'ketogenic diet': ['ketogenic diet', 'keto diet', 'high fat low carb'],
            'low glucose diet': ['low glucose diet', 'low sugar diet', 'glucose restriction'],
            'low carb diet': ['low carb diet', 'low carbohydrate diet', 'carb restriction'],
            'mediterranean diet': ['mediterranean diet'],
            'intermittent fasting': ['intermittent fasting', 'time restricted eating'],
            
            # Cancer-related compounds
            'cancer remission': ['cancer remission', 'tumor regression', 'cancer regression'],
            'cancer cells': ['cancer cells', 'tumor cells', 'malignant cells'],
            'cancer treatment': ['cancer treatment', 'cancer therapy', 'oncology treatment'],
            'cancer prevention': ['cancer prevention', 'cancer risk reduction'],
            
            # Metabolic compounds
            'glucose deprivation': ['glucose deprivation', 'glucose restriction', 'sugar restriction'],
            'insulin resistance': ['insulin resistance', 'metabolic dysfunction'],
            'blood glucose': ['blood glucose', 'blood sugar', 'glycemia'],
            
            # Heart/Cardiovascular compounds
            'heart disease': ['heart disease', 'cardiovascular disease', 'coronary artery disease', 'cvd'],
            'heart muscle function': ['heart muscle function', 'cardiac function', 'myocardial function'],
            'blood flow': ['blood flow', 'circulation', 'vascular function'],
            'oxidative stress': ['oxidative stress', 'free radical damage', 'antioxidant'],
            'blood pressure': ['blood pressure', 'hypertension', 'bp'],
            
            # Supplements and Vitamins (MAJOR EXPANSION)
            'coenzyme q10': ['coenzyme q10', 'co-enzyme q10', 'coq10', 'ubiquinone', 'ubiquinol'],
            'vitamin d': ['vitamin d', 'vitamin d3', 'cholecalciferol', 'calciferol'],
            'vitamin c': ['vitamin c', 'ascorbic acid', 'ascorbate'],
            'vitamin e': ['vitamin e', 'tocopherol', 'alpha-tocopherol'],
            'vitamin k': ['vitamin k', 'vitamin k2', 'menaquinone'],
            'vitamin b12': ['vitamin b12', 'cobalamin', 'cyanocobalamin'],
            'folate': ['folate', 'folic acid', 'vitamin b9'],
            'omega 3 fatty acids': ['omega 3', 'omega-3', 'fish oil', 'epa', 'dha', 'omega 3 fatty acids'],
            'magnesium': ['magnesium', 'magnesium supplement', 'mg supplement'],
            'zinc': ['zinc', 'zinc supplement'],
            'selenium': ['selenium', 'selenium supplement'],
            'iron': ['iron', 'iron supplement', 'ferrous sulfate'],
            'calcium': ['calcium', 'calcium supplement'],
            'probiotics': ['probiotics', 'probiotic', 'beneficial bacteria', 'lactobacillus'],
            'turmeric': ['turmeric', 'curcumin', 'curcuma longa'],
            'garlic': ['garlic', 'allicin', 'garlic extract'],
            'ginger': ['ginger', 'zingiber officinale'],
            'green tea': ['green tea', 'green tea extract', 'egcg', 'catechins'],
            'resveratrol': ['resveratrol', 'red wine extract'],
            'quercetin': ['quercetin', 'flavonoid'],
            'alpha lipoic acid': ['alpha lipoic acid', 'lipoic acid', 'ala'],
            'acetyl l carnitine': ['acetyl l carnitine', 'alcar', 'carnitine'],
            'creatine': ['creatine', 'creatine monohydrate'],
            'glucosamine': ['glucosamine', 'glucosamine sulfate'],
            'chondroitin': ['chondroitin', 'chondroitin sulfate'],
            'saw palmetto': ['saw palmetto', 'serenoa repens'],
            'milk thistle': ['milk thistle', 'silymarin'],
            'echinacea': ['echinacea', 'purple coneflower'],
            'ginkgo biloba': ['ginkgo biloba', 'ginkgo'],
            'ginseng': ['ginseng', 'panax ginseng'],
            'rhodiola': ['rhodiola', 'rhodiola rosea'],
            'ashwagandha': ['ashwagandha', 'withania somnifera'],
            'st johns wort': ['st johns wort', 'hypericum perforatum'],
            'valerian': ['valerian', 'valerian root'],
            'melatonin': ['melatonin', 'melatonin supplement'],
            
            # Treatment compounds
            'chemotherapy resistance': ['chemotherapy resistance', 'chemo resistance'],
            'radiation therapy': ['radiation therapy', 'radiotherapy'],
            'immunotherapy': ['immunotherapy', 'immune therapy'],
            
            # Condition compounds
            'diabetes': ['diabetes', 'diabetes mellitus', 'type 2 diabetes', 'diabetic'],
            'arthritis': ['arthritis', 'rheumatoid arthritis', 'osteoarthritis'],
            'alzheimers disease': ['alzheimers', 'alzheimer disease', 'dementia'],
            'parkinsons disease': ['parkinsons', 'parkinson disease'],
            'depression': ['depression', 'major depression', 'depressive disorder'],
            'anxiety': ['anxiety', 'anxiety disorder', 'generalized anxiety'],
            'inflammation': ['inflammation', 'inflammatory response'],
            'immune system': ['immune system', 'immunity', 'immune function'],
        }
        
        # Find compound terms in the claim
        found_compounds = []
        for main_term, variations in compound_medical_terms.items():
            for variation in variations:
                if variation in claim_lower:
                    found_compounds.append(main_term)
                    break  # Found this compound, move to next
        
        # Extract relationship words that indicate how concepts connect
        relationship_words = []
        relationship_patterns = [
            r'\b(cause|causes|causing|lead to|leads to|result in|results in)\b',
            r'\b(prevent|prevents|preventing|reduce|reduces|reducing)\b',
            r'\b(treat|treats|treating|cure|cures|curing)\b',
            r'\b(improve|improves|improving|enhance|enhances)\b',
            r'\b(following|using|taking|consuming|eating)\b'
        ]
        
        for pattern in relationship_patterns:
            matches = re.findall(pattern, claim_lower)
            relationship_words.extend(matches)
        
        # Add dynamic supplement detection
        dynamic_supplements = self._detect_supplements_dynamically(claim_text)

        # Combine dictionary-found compounds with dynamically detected ones
        all_compounds = found_compounds + dynamic_supplements

        # Remove duplicates while preserving order
        seen = set()
        unique_compounds = []
        for compound in all_compounds:
            if compound.lower() not in seen:
                unique_compounds.append(compound)
                seen.add(compound.lower())

        return {
            'compound_terms': unique_compounds,
            'dictionary_terms': found_compounds,
            'dynamic_terms': dynamic_supplements,
            'relationships': relationship_words,
            'has_medical_content': len(unique_compounds) > 0
        }

    def _detect_supplements_dynamically(self, claim_text):
        """
        Dynamically detect supplement/drug names with focused precision
        """
        claim_lower = claim_text.lower()
        detected_supplements = []
        
        # FOCUSED supplement patterns (much more precise)
        supplement_patterns = [
            # Specific supplement name patterns
            r'\b(co-?enzyme\s+q10|coq10|ubiquinone)\b',
            r'\b(alpha[-\s]?lipoic\s+acid|ala)\b',
            r'\b(acetyl[-\s]?l[-\s]?carnitine|alcar)\b',
            r'\b(omega[-\s]?[36]|fish\s+oil|epa|dha)\b',
            r'\b(vitamin\s+[a-k][0-9]*|b[-\s]?complex)\b',
            r'\b(glucosamine|chondroitin|msm)\b',
            r'\b(curcumin|turmeric)\b',
            r'\b(resveratrol|quercetin)\b',
            r'\b(probiotics?|lactobacillus|bifidobacterium)\b',
            r'\b(melatonin|magnesium|zinc|selenium)\b',
            
            # Simple supplement + word combinations (limited)
            r'\b([a-z]{4,12})\s+supplement\b',
            r'\b([a-z]{4,12})\s+extract\b',
        ]
        
        for pattern in supplement_patterns:
            matches = re.findall(pattern, claim_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(filter(None, match)).strip()
                else:
                    match = match.strip()
                
                # Only add if it's a reasonable length and not a common word
                if match and 3 <= len(match) <= 25:
                    detected_supplements.append(match)
        
        return list(set(detected_supplements))
        
        for pattern in supplement_patterns:
            matches = re.findall(pattern, claim_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match).strip()
                else:
                    match = match.strip()
                
                # Filter out common words that aren't supplements
                exclude_words = ['the', 'and', 'for', 'with', 'that', 'this', 'are', 'is', 'can', 'may', 'will']
                if match and match not in exclude_words and len(match) > 2:
                    detected_supplements.append(match)
        
        # Remove duplicates
        return list(set(detected_supplements))

    def _detect_dietary_interventions(self, claim_text):
        """
        Detect dietary interventions and convert them to medical search terms
        """
        claim_lower = claim_text.lower()
        dietary_interventions = []
        
        # Dietary intervention patterns
        dietary_patterns = {
            
            # ADD THIS NEW FIBER SECTION AT THE BEGINNING
            'fiber supplementation': [
                r'psyllium(?:\s+husk)?',
                r'fiber\s+supplement',
                r'dietary\s+fiber',
                r'soluble\s+fiber',
                r'insoluble\s+fiber',
                r'bulk[-\s]?forming\s+(?:agent|laxative)',
                r'methylcellulose',
                r'wheat\s+bran',
                r'oat\s+fiber'
            ],
            
            'laxative therapy': [
                r'laxative',
                r'stool\s+softener',
                r'bulk[-\s]?forming',
                r'osmotic\s+(?:agent|laxative)',
                r'stimulant\s+laxative'
            ],
            
            # Elimination diets
            'plant based diet': [
                r'stop(?:ping)?\s+(?:consumption\s+of\s+|eating\s+)?(?:meat|dairy|animal products)',
                r'avoid(?:ing)?\s+(?:meat|dairy|animal products)',
                r'eliminat(?:e|ing)\s+(?:meat|dairy|animal products)',
                r'no\s+(?:meat|dairy|animal products)',
                r'vegan|plant[-\s]?based'
            ],
            
            'low fat diet': [
                r'stop(?:ping)?\s+(?:consumption\s+of\s+|eating\s+)?(?:butter|cream|cheese|fatty foods)',
                r'avoid(?:ing)?\s+(?:butter|cream|cheese|fatty foods)',
                r'eliminat(?:e|ing)\s+(?:butter|cream|cheese|fatty foods)',
                r'low[-\s]?fat',
                r'reduc(?:e|ing)\s+(?:fat|butter|cream)'
            ],
            
            'glucose restriction': [
                r'reduc(?:e|ing)\s+(?:glucose|sugar|carbohydrate)',
                r'restrict(?:ing)?\s+(?:glucose|sugar|carbohydrate)',
                r'low[-\s]?glucose',
                r'low[-\s]?sugar',
                r'low[-\s]?carb(?:ohydrate)?',
                r'avoid(?:ing)?\s+(?:glucose|sugar|carbohydrate)',
                r'eliminat(?:e|ing)\s+(?:glucose|sugar|carbohydrate)'
            ],
            
            'ketogenic diet': [
                r'keto(?:genic)?',
                r'high[-\s]?fat\s+low[-\s]?carb',
                r'ketosis'
            ],
            
            'mediterranean diet': [
                r'mediterranean',
                r'olive\s+oil\s+based'
            ],
            
            'intermittent fasting': [
                r'intermittent\s+fasting',
                r'time[-\s]?restricted\s+eating',
                r'fasting'
            ]
        }
        
        for intervention_name, patterns in dietary_patterns.items():
            for pattern in patterns:
                if re.search(pattern, claim_lower):
                    dietary_interventions.append(intervention_name)
                    break  # Found this intervention, move to next
        
        return list(set(dietary_interventions))
    
    def _detect_medical_outcomes(self, claim_text):
        """
        Detect medical outcomes and conditions mentioned in claims
        """
        claim_lower = claim_text.lower()
        medical_outcomes = []
        
        # Medical outcome patterns
        # Medical outcome patterns
        outcome_patterns = {
            'cholesterol reduction': [
                r'cholesterol\s+levels?\s+(?:came\s+back\s+to\s+normal|decreased|reduced|lowered)',
                r'lower(?:ed|ing)?\s+cholesterol',
                r'reduc(?:e|ed|ing)\s+cholesterol',
                r'cholesterol\s+(?:reduction|decrease)'
            ],
            
            # ADD THESE NEW FIBER-RELATED PATTERNS
            'constipation relief': [
                r'constipation',
                r'move\s+things\s+along',
                r'bowel\s+movement',
                r'stool\s+(?:passage|movement)',
                r'difficulty\s+moving\s+stool',
                r'hard\s+stool',
                r'intestinal\s+transit'
            ],
            
            'stool consistency': [
                r'stool\s+consistency',
                r'soft(?:en|er)?\s+stool',
                r'bulk(?:ing)?\s+(?:agent|fiber)',
                r'water\s+(?:in|into)\s+(?:the\s+)?(?:stool|colon)',
                r'retain\s+water',
                r'absorb\s+water',
                r'stool\s+(?:becomes|is)\s+(?:dehydrated|hard|dry)'
            ],
            
            'digestive health': [
                r'digestive\s+health',
                r'gut\s+health',
                r'intestinal\s+health',
                r'colon\s+function',
                r'gastrointestinal',
                r'gi\s+(?:tract|function)'
            ],
            
            'transit time': [
                r'transit\s+time',
                r'intestinal\s+transit',
                r'colonic\s+transit',
                r'gut\s+transit',
                r'move\s+through\s+(?:the\s+)?intestines?',
                r'speed\s+(?:up|through)\s+(?:the\s+)?(?:gut|intestines?|colon)'
            ],
            
            # EXISTING PATTERNS (keep these)
            'cancer remission': [
                r'cancer\s+(?:remission|regression)',
                r'tumor\s+(?:shrinkage|regression)',
                r'cancer\s+(?:cells\s+)?(?:died|destroyed|eliminated)'
            ],
            
            'blood pressure reduction': [
                r'blood\s+pressure\s+(?:reduced|lowered|decreased)',
                r'lower(?:ed|ing)?\s+blood\s+pressure',
                r'hypertension\s+(?:improved|resolved)'
            ],
            
            'diabetes improvement': [
                r'diabetes\s+(?:reversed|improved|resolved)',
                r'blood\s+sugar\s+(?:normalized|controlled|improved)',
                r'insulin\s+(?:sensitivity\s+)?improved'
            ],
            
            'heart disease prevention': [
                r'heart\s+disease\s+(?:prevention|reduced\s+risk)',
                r'cardiovascular\s+(?:health|protection)',
                r'cardiac\s+(?:function\s+)?improved'
            ],
            
            'inflammation reduction': [
                r'reduc(?:e|ed|ing)\s+inflammation',
                r'anti[-\s]?inflammatory',
                r'inflammation\s+(?:decreased|lowered)'
            ]
        }
        
        for outcome_name, patterns in outcome_patterns.items():
            for pattern in patterns:
                if re.search(pattern, claim_lower):
                    medical_outcomes.append(outcome_name)
                    break
        
        return list(set(medical_outcomes))

    def _generate_smart_medical_queries(self, claim_text):
        """
        Generate precise, focused medical search queries for PubMed
        """
        # Get all types of medical concepts
        concepts = self._extract_medical_concepts_advanced(claim_text)
        dietary_interventions = self._detect_dietary_interventions(claim_text)
        medical_outcomes = self._detect_medical_outcomes(claim_text)
        
        queries = []
        
        # Strategy 1: Dietary Intervention + Medical Outcome (HIGHEST PRIORITY)
        for intervention in dietary_interventions:
            for outcome in medical_outcomes:
                # Create focused medical queries
                queries.append(f'"{intervention}" AND "{outcome}"')
                queries.append(f'"{intervention}" AND "{outcome}" AND (clinical trial OR systematic review)')
                
        # Strategy 2: Intervention + Specific Condition
        condition_terms = ['cholesterol', 'cancer', 'diabetes', 'heart disease', 'hypertension']
        for intervention in dietary_interventions:
            for condition in condition_terms:
                if condition in claim_text.lower():
                    queries.append(f'"{intervention}" AND {condition}')
                    
        # Strategy 3: Mechanism-based queries for biological claims
        claim_lower = claim_text.lower()
        
        # For glucose/cancer claims
        if 'glucose' in claim_lower and 'cancer' in claim_lower:
            queries.extend([
                '"glucose restriction" AND "cancer cells"',
                '"glucose metabolism" AND cancer',
                '"ketogenic diet" AND cancer',
                '"low glucose diet" AND "tumor growth"'
            ])
        
        # For cholesterol/diet claims  
        if 'cholesterol' in claim_lower and any(food in claim_lower for food in ['meat', 'dairy', 'butter', 'cheese']):
            queries.extend([
                '"plant based diet" AND cholesterol',
                '"vegan diet" AND "cholesterol levels"',
                '"dietary cholesterol" AND "serum cholesterol"',
                '"saturated fat" AND cholesterol'
            ])
        
        # Strategy 4: Supplement + Condition (if supplements detected)
        compound_terms = concepts.get('compound_terms', [])
        supplement_terms = [term for term in compound_terms if any(supp in term.lower() 
                        for supp in ['coenzyme', 'vitamin', 'omega', 'extract', 'supplement'])]
        
        for supplement in supplement_terms:
            # Find relevant conditions in the claim
            if 'heart' in claim_lower:
                queries.append(f'"{supplement}" AND "cardiovascular disease"')
                queries.append(f'"{supplement}" AND "heart function"')
            if 'cholesterol' in claim_lower:
                queries.append(f'"{supplement}" AND cholesterol')
            if 'cancer' in claim_lower:
                queries.append(f'"{supplement}" AND cancer')
                
        # Strategy 5: Focused single terms (fallback)
        if not queries:
            # Extract the most important medical terms
            important_terms = dietary_interventions + medical_outcomes + supplement_terms
            
            for term in important_terms[:3]:  # Limit to top 3
                queries.append(f'"{term}"')
                queries.append(f'"{term}" AND (systematic review OR meta-analysis)')
        
        # Clean and validate queries before deduplication
        cleaned_queries = []
        for query in queries:
            cleaned_query = self._clean_and_validate_query(query)
            if cleaned_query:
                cleaned_queries.append(cleaned_query)

        # Remove duplicates and limit to 8 queries
        seen = set()
        unique_queries = []
        for query in cleaned_queries:
            if query.lower() not in seen:
                unique_queries.append(query)
                seen.add(query.lower())

        # Add fallback fiber queries if the original claim was about fiber and we have few results
        original_text_lower = claim_text.lower()
        if (len(unique_queries) < 3 and 
            any(term in original_text_lower for term in ['fiber', 'psyllium', 'stool', 'constipation', 'bowel'])):
            
            fallback_queries = [
                '"dietary fiber" AND constipation',
                '"psyllium" AND "stool consistency"',
                '"bulk forming laxative" AND "intestinal transit"',
                '"soluble fiber" AND "water absorption"',
                'fiber AND "colonic function"'
            ]
            
            for fallback in fallback_queries:
                if fallback.lower() not in seen:
                    unique_queries.append(fallback)
                    seen.add(fallback.lower())
                    if len(unique_queries) >= 8:
                        break

        return unique_queries[:8]

    def _generate_basic_fallback_queries(self, claim_text):
        """
        Fallback query generation when compound terms aren't detected
        """
        # Extract basic keywords
        medical_keywords = []
        basic_terms = ['cancer', 'diabetes', 'heart', 'diet', 'treatment', 'therapy', 
                    'vitamin', 'supplement', 'exercise', 'nutrition', 'glucose', 'insulin']
        
        claim_lower = claim_text.lower()
        for term in basic_terms:
            if term in claim_lower:
                medical_keywords.append(term)
        
        queries = []
        if len(medical_keywords) >= 2:
            # Create simple combinations
            queries.append(f'"{medical_keywords[0]}" AND "{medical_keywords[1]}"')
            
        # Add individual term searches
        for term in medical_keywords[:3]:
            queries.append(f'"{term}" AND evidence')
            
        return queries[:5]

    def generate_pubmed_queries_with_gpt(self, claim_text, api_key, max_queries=5):
        """
        Use GPT to generate optimal PubMed search queries from a health claim
        """
        try:
            if not api_key:
                return []
                
            prompt = f'''Given this health claim, generate up to {max_queries} optimal PubMed search queries.
            
            Rules:
            1. PRIORITIZE specific drug names if mentioned (e.g., "evacetrapib", "ezetimibe", "atorvastatin")
            2. Use medical terminology and MeSH terms when appropriate
            3. Keep queries concise (2-6 words ideal for simple queries, longer for complex Boolean)
            4. Use AND/OR operators strategically
            5. Include both specific drug names AND their therapeutic classes
            6. Focus on intervention-outcome relationships
            7. Use quotes for exact phrases when needed
            8. For pharmaceutical claims, include drug name + efficacy/safety/outcomes
            9. Return only the queries, one per line, no numbering or explanations
            
            Health claim: {claim_text}
            
            Example output formats:
            For drug claims: "evacetrapib" AND "cardiovascular outcomes"
            For supplements: "plant sterols" AND "cholesterol absorption"
            For combinations: "ezetimibe" AND "statins"
            Drug class: "CETP inhibitors" AND "clinical trials"'''
            
            # Prepare the API request
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a medical research expert specializing in PubMed search query optimization. Generate precise, effective search queries."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            # Make API call
            response = make_openai_request(payload, api_key)
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content'].strip()
                
                # Parse the response into individual queries
                queries = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and len(line) > 3:
                        # Remove any numbering (1., -, etc.)
                        cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                        cleaned_line = re.sub(r'^[\-\*]\s*', '', cleaned_line)
                        
                        if cleaned_line:
                            queries.append(cleaned_line)
                
                # Validate and filter queries
                validated_queries = self._validate_search_queries(queries, claim_text)
                
                self._log_query_generation("GPT", len(validated_queries), "Success" if validated_queries else "No valid queries", validated_queries)
                return validated_queries[:max_queries]
            
            self._log_query_generation("GPT", 0, "API response invalid")
            return []
            
        except Exception as e:
            self._log_query_generation("GPT", 0, f"Error: {str(e)}")
            return []
    
    def _validate_search_queries(self, queries, original_claim):
        """
        Validate search queries for quality and relevance
        """
        validated = []
        
        # Medical stop words that shouldn't be queries by themselves
        stop_words = {
            'the', 'and', 'or', 'not', 'for', 'with', 'by', 'from', 'to', 'of', 'in', 'on', 'at',
            'study', 'research', 'trial', 'effect', 'effects', 'result', 'results', 'analysis'
        }
        
        for query in queries:
            if not query or len(query.strip()) < 3:
                continue
                
            query_clean = query.strip()
            
            # Check if query is just stop words
            query_words = re.findall(r'\b\w+\b', query_clean.lower())
            if all(word in stop_words for word in query_words):
                continue
            
            # Check for balanced quotes and parentheses
            if query_clean.count('"') % 2 != 0:
                continue
            if query_clean.count('(') != query_clean.count(')'):
                continue
            
            # Check length (not too short, not too long)
            if len(query_clean) < 5 or len(query_clean) > 200:
                continue
            
            # Check if query contains at least one medical/scientific/cosmetic/pharmaceutical term
            medical_indicators = [
                r'\b(?:and|or)\b',  # Boolean operators
                r'"[^"]{3,}"',      # Quoted phrases
                r'\b\w+tion\b',     # -tion words (often medical)
                r'\b\w+ment\b',     # -ment words
                r'\b\w+ine\b',      # -ine words (often chemicals)
                r'\b\w+ol\b',       # -ol words (often chemicals)
                r'\b(?:meta|systematic|randomized|controlled|clinical|double|blind|placebo)\b',
                
                # Pharmaceutical drug indicators (HIGH PRIORITY)
                r'\b\w*(?:statin|pril|sartan|olol|ipine|afib|mab|ine|ide|ib)\b',  # Drug suffixes
                r'\b(?:evacetrapib|ezetimibe|atorvastatin|rosuvastatin|simvastatin)\b',  # Specific drugs
                r'\b(?:metformin|sitagliptin|empagliflozin|semaglutide|liraglutide)\b',
                r'\b(?:alirocumab|evolocumab|inclisiran|bempedoic)\b',
                r'\b(?:efficacy|safety|outcomes|cardiovascular|cholesterol|lipid)\b',  # Drug study terms
                r'\b(?:inhibitor|agonist|antagonist|receptor|enzyme)\b',  # Pharmacological terms
                
                # Skincare/cosmetic indicators
                r'\b(?:acid|acids)\b',  # Common in skincare
                r'\b(?:skin|facial|topical|dermal)\b',
                r'\b(?:cleanser|serum|cream|lotion|treatment)\b',
                r'\b(?:acne|wrinkles|aging|pigmentation|pores)\b',
                r'\b(?:salicylic|glycolic|hyaluronic|retinol|niacinamide)\b',
                r'\b(?:moisturiz|hydrat|exfoliat)\b',  # Word stems
                r'\b(?:anti|pro)\w+\b'  # Anti-aging, pro-collagen, etc.
            ]
            
            has_medical_term = any(re.search(pattern, query_clean.lower()) for pattern in medical_indicators)
            
            # Check if query has some relevance to original claim (lowered threshold)
            claim_words = set(re.findall(r'\b\w{3,}\b', original_claim.lower()))
            query_words = set(re.findall(r'\b\w{3,}\b', query_clean.lower()))
            relevance_score = len(claim_words.intersection(query_words)) / max(len(claim_words), 1)
            
            # More lenient validation - accept if it has medical terms OR decent relevance
            if has_medical_term or relevance_score > 0.05:  # Lowered from 0.1 to 0.05
                validated.append(query_clean)
        
        return validated
    
    def enable_query_logging(self, enabled=True):
        """
        Enable or disable query generation logging for debugging
        """
        self._enable_logging = enabled
        if enabled:
            print("[PubMed Query Gen] Logging enabled - you'll see query generation details")
        else:
            print("[PubMed Query Gen] Logging disabled")
    
    def _log_query_generation(self, method, query_count, status, queries=None):
        """
        Log query generation attempts for debugging
        """
        if hasattr(self, '_enable_logging') and self._enable_logging:
            timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] PubMed Query Gen - Method: {method}, Count: {query_count}, Status: {status}")
            
            if queries and len(queries) > 0:
                print(f"  Generated queries:")
                for i, query in enumerate(queries[:3], 1):  # Show first 3 queries
                    print(f"    {i}. {query}")
                if len(queries) > 3:
                    print(f"    ... and {len(queries) - 3} more")
            print()  # Add blank line for readability
    
    def _generate_focused_search_queries(self, claim_text, api_key=None):
        """
        Generate focused, high-quality PubMed search queries using enhanced fallback hierarchy
        """
        # Enhanced fallback hierarchy:
        # 1. GPT-powered generation (if API key available)
        # 2. Smart medical concept extraction
        # 3. Basic keyword extraction  
        # 4. Truncated claim text
        
        # Preprocess claim to extract core components from marketing language
        core_claim = self._extract_core_claim(claim_text)
        
        final_queries = []
        
        # First: Try GPT-based query generation (use core claim if available)
        if api_key:
            query_text = core_claim if core_claim != claim_text else claim_text
            gpt_queries = self.generate_pubmed_queries_with_gpt(query_text, api_key, max_queries=5)
            if gpt_queries:
                final_queries.extend(gpt_queries)
                self._log_query_generation("GPT", len(gpt_queries), "Success", gpt_queries)
                return final_queries[:5]
        
        # Second: Use smart medical query generation (try both core and original)
        smart_queries = self._generate_smart_medical_queries(core_claim)
        if not smart_queries and core_claim != claim_text:
            smart_queries = self._generate_smart_medical_queries(claim_text)
            
        if smart_queries:
            validated_smart = self._validate_search_queries(smart_queries, claim_text)
            if validated_smart:
                final_queries.extend(validated_smart)
                self._log_query_generation("Smart", len(validated_smart), "Success")
                return final_queries[:5]
        
        # Third: Priority extraction for drug names (most important for pharma claims)
        drug_names = self._extract_drug_names(claim_text)
        if drug_names:
            drug_queries = []
            
            for drug in drug_names[:2]:  # Focus on top 2 drugs
                # Add exact drug name query
                drug_queries.append(f'"{drug}"')
                
                # Add drug with common outcomes
                common_outcomes = ['efficacy', 'safety', 'clinical trial', 'cardiovascular', 'outcomes']
                for outcome in common_outcomes[:2]:
                    drug_queries.append(f'"{drug}" AND {outcome}')
                
                # Add drug class context if it's a known drug
                drug_class = self._get_drug_class(drug)
                if drug_class and drug_class != drug:
                    drug_queries.append(f'"{drug}" AND "{drug_class}"')
            
            validated_drugs = self._validate_search_queries(drug_queries, claim_text)
            if validated_drugs:
                final_queries.extend(validated_drugs)
                self._log_query_generation("Drugs", len(validated_drugs), "Success - Found specific drugs")
                return final_queries[:5]
        
        # Fourth: Use basic intervention-outcome extraction
        components = self._extract_intervention_and_outcome(claim_text)
        basic_queries = []
        
        if components['interventions'] and components['outcomes']:
            for intervention in components['interventions'][:2]:
                for outcome in components['outcomes'][:2]:
                    query = f'"{intervention}" AND "{outcome}"'
                    basic_queries.append(query)
        
        # Add single-term queries from key concepts
        if components['interventions']:
            for intervention in components['interventions'][:2]:
                basic_queries.append(f'"{intervention}"')
        
        if basic_queries:
            validated_basic = self._validate_search_queries(basic_queries, claim_text)
            if validated_basic:
                final_queries.extend(validated_basic)
                self._log_query_generation("Basic", len(validated_basic), "Success")
                return final_queries[:5]
        
        # Fourth: Extract key medical terms using regex (try core claim first)
        medical_terms = self._extract_medical_terms_robust(core_claim)
        if not medical_terms and core_claim != claim_text:
            medical_terms = self._extract_medical_terms_robust(claim_text)
            
        if medical_terms:
            term_queries = [f'"{term}"' for term in medical_terms[:3]]
            
            # Add logical combinations for skincare terms
            for term in medical_terms[:2]:
                if 'acid' in term:
                    # Add common skincare combinations
                    term_queries.extend([
                        f'"{term}" AND acne',
                        f'"{term}" AND "skin care"',
                        f'"{term}" AND treatment'
                    ])
                elif term in ['cleanser', 'serum', 'moisturizer']:
                    term_queries.extend([
                        f'"{term}" AND "skin care"',
                        f'"{term}" AND dermatology'
                    ])
            
            validated_terms = self._validate_search_queries(term_queries, claim_text)
            if validated_terms:
                final_queries.extend(validated_terms)
                self._log_query_generation("Terms", len(validated_terms), "Success")
                return final_queries[:5]
        
        # Fifth: Simple keyword extraction as emergency fallback
        keywords = self._extract_simple_keywords(claim_text)
        if keywords:
            keyword_queries = []
            # Create simple quoted queries from keywords
            for keyword in keywords[:3]:
                keyword_queries.append(f'"{keyword}"')
            
            # Create combinations for better results
            if len(keywords) >= 2:
                keyword_queries.append(f'"{keywords[0]}" AND "{keywords[1]}"')
            
            if keyword_queries:
                self._log_query_generation("Keywords", len(keyword_queries), "Using simple keywords")
                return keyword_queries[:5]
        
        # Last resort: Use truncated claim text (max 50 characters)
        fallback_query = claim_text[:50].strip()
        if len(fallback_query) > 10:
            self._log_query_generation("Fallback", 1, "Using truncated claim")
            return [fallback_query]
        
        # Ultimate fallback
        self._log_query_generation("Ultimate", 1, "Using generic health term")
        return ["health intervention clinical trial"]
    
    def _extract_simple_keywords(self, claim_text):
        """
        Extract simple keywords as a last resort for query generation
        """
        claim_lower = claim_text.lower()
        
        # Remove common stop words and marketing terms
        stop_words = {
            'start', 'with', 'this', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'as', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'since', 'without', 'under', 'over', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now', 'gentle', 'mild', 'new', 'best', 'great'
        }
        
        # Extract words, remove stop words
        words = re.findall(r'\b\w{3,}\b', claim_lower)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Prioritize medical/cosmetic terms
        priority_terms = [
            'acid', 'vitamin', 'oil', 'extract', 'cleanser', 'serum', 'cream', 'treatment',
            'supplement', 'therapy', 'medication', 'drug', 'health', 'skin', 'care'
        ]
        
        # Sort keywords - priority terms first, then by length
        def sort_key(word):
            if any(term in word for term in priority_terms):
                return (0, -len(word))  # Priority terms first, longer first
            return (1, -len(word))  # Non-priority terms second, longer first
        
        keywords.sort(key=sort_key)
        
        return keywords[:5]
    
    def _extract_drug_names(self, claim_text):
        """
        Extract specific drug names using pharmaceutical nomenclature patterns
        """
        text_lower = claim_text.lower()
        drug_names = []
        
        # Pharmaceutical naming patterns - these catch drug names by their structure
        pharma_patterns = [
            # Specific known drugs (high priority)
            r'\b(evacetrapib|ezetimibe|atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin|fluvastatin|pitavastatin)\b',
            r'\b(fenofibrate|gemfibrozil|cholestyramine|colesevelam|alirocumab|evolocumab|inclisiran)\b',
            r'\b(metformin|glipizide|glyburide|pioglitazone|rosiglitazone|sitagliptin|linagliptin)\b',
            r'\b(empagliflozin|canagliflozin|semaglutide|liraglutide|dulaglutide)\b',
            r'\b(bempedoic\s+acid)\b',
            
            # Generic drug suffixes (pharmaceutical nomenclature)
            r'\b(\w{4,}(?:statin))\b',        # *statin (cholesterol drugs)
            r'\b(\w{4,}(?:pril))\b',          # *pril (ACE inhibitors)
            r'\b(\w{4,}(?:sartan))\b',        # *sartan (ARBs)
            r'\b(\w{4,}(?:olol))\b',          # *olol (beta blockers)
            r'\b(\w{4,}(?:ipine))\b',         # *ipine (calcium channel blockers)
            r'\b(\w{4,}(?:afib))\b',          # *afib (newer compounds)
            r'\b(\w{4,}(?:mab))\b',           # *mab (monoclonal antibodies)
            r'\b(\w{6,}(?:ine|ide|ib))\b',    # *ine, *ide, *ib (various drug classes)
            
            # Brand name patterns (often capitalized in text)
            r'\b([A-Z][a-z]{3,}(?:ol|ex|in|ax|or|et|um|an))\b',  # Common brand name endings
        ]
        
        for pattern in pharma_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    drug_name = ' '.join(match).strip()
                else:
                    drug_name = match.strip()
                
                # Filter out common false positives
                false_positives = {
                    'protein', 'collagen', 'elastin', 'keratin', 'melanin', 'insulin',
                    'vitamin', 'mineral', 'calcium', 'sodium', 'carbon', 'oxygen',
                    'morning', 'evening', 'during', 'within', 'obtain', 'explain'
                }
                
                if (len(drug_name) >= 4 and 
                    drug_name not in false_positives and 
                    drug_name not in drug_names and
                    not drug_name.isdigit()):
                    drug_names.append(drug_name)
        
        return drug_names[:5]  # Return top 5 drug names
    
    def _get_drug_class(self, drug_name):
        """
        Map specific drug names to their therapeutic classes
        """
        drug_classes = {
            # Statins (HMG-CoA reductase inhibitors)
            'atorvastatin': 'statins',
            'rosuvastatin': 'statins', 
            'simvastatin': 'statins',
            'pravastatin': 'statins',
            'lovastatin': 'statins',
            'fluvastatin': 'statins',
            'pitavastatin': 'statins',
            
            # CETP inhibitors
            'evacetrapib': 'CETP inhibitors',
            'dalcetrapib': 'CETP inhibitors',
            'anacetrapib': 'CETP inhibitors',
            
            # Cholesterol absorption inhibitors
            'ezetimibe': 'cholesterol absorption inhibitors',
            
            # Fibrates
            'fenofibrate': 'fibrates',
            'gemfibrozil': 'fibrates',
            
            # Bile acid sequestrants
            'cholestyramine': 'bile acid sequestrants',
            'colesevelam': 'bile acid sequestrants',
            
            # PCSK9 inhibitors
            'alirocumab': 'PCSK9 inhibitors',
            'evolocumab': 'PCSK9 inhibitors',
            'inclisiran': 'PCSK9 inhibitors',
            
            # ATP citrate lyase inhibitors
            'bempedoic acid': 'ATP citrate lyase inhibitors',
            
            # Diabetes medications
            'metformin': 'biguanides',
            'glipizide': 'sulfonylureas',
            'glyburide': 'sulfonylureas',
            'pioglitazone': 'thiazolidinediones',
            'rosiglitazone': 'thiazolidinediones',
            'sitagliptin': 'DPP-4 inhibitors',
            'linagliptin': 'DPP-4 inhibitors',
            'empagliflozin': 'SGLT2 inhibitors',
            'canagliflozin': 'SGLT2 inhibitors',
            'semaglutide': 'GLP-1 agonists',
            'liraglutide': 'GLP-1 agonists',
            'dulaglutide': 'GLP-1 agonists',
        }
        
        return drug_classes.get(drug_name.lower(), None)
    
    def _extract_medical_terms_robust(self, claim_text):
        """
        Extract medical terms using robust pattern matching - enhanced for skincare/cosmetic claims
        """
        text_lower = claim_text.lower()
        medical_terms = []
        
        # Enhanced patterns for medical terms - now including skincare/cosmetic
        patterns = [
            # Compound medical terms (2-3 words)
            r'\b(?:blood|serum|plasma)\s+(?:pressure|cholesterol|glucose|sugar|lipids?)\b',
            r'\b(?:heart|cardiovascular|cardiac)\s+(?:disease|health|risk|function)\b',
            r'\b(?:plant|dietary|soluble|insoluble)\s+(?:sterols?|stanols?|fiber|protein)\b',
            r'\b(?:omega|fatty)\s+(?:acids?|3|6)\b',
            r'\b(?:vitamin|mineral)\s+[a-zA-Z]\d?\b',
            r'\b(?:meta|systematic)\s+(?:analysis|review)\b',
            
            # Skincare/Dermatological compounds
            r'\b(?:salicylic|glycolic|lactic|hyaluronic|retinoic)\s+(?:acid|acids?)\b',
            r'\b(?:alpha|beta)\s+(?:hydroxy|acids?)\b',
            r'\b(?:skin|facial|topical)\s+(?:cleanser|treatment|care|therapy)\b',
            r'\b(?:anti)\s*(?:aging|acne|inflammatory|bacterial)\b',
            
            # Pharmaceutical drug name patterns
            r'\b\w*(?:statin|pril|sartan|olol|ipine|afib|mab|ine|ide|ib)\b',  # Common drug suffixes
            r'\b(?:evacetrapib|ezetimibe|atorvastatin|rosuvastatin|simvastatin|pravastatin)\b',
            r'\b(?:metformin|sitagliptin|empagliflozin|semaglutide|liraglutide)\b',
            r'\b(?:alirocumab|evolocumab|inclisiran|bempedoic\s+acid)\b',
            
            # Single medical/cosmetic terms
            r'\b(?:phytosterols?|phytostanols?|psyllium|niacin|statins?|fibrates?)\b',
            r'\b(?:cholesterol|triglycerides?|lipoproteins?|ldl|hdl|vldl)\b',
            r'\b(?:diabetes|hypertension|obesity|metabolic)\b',
            r'\b(?:randomized|controlled|clinical|double|blind|placebo)\b',
            
            # Skincare active ingredients
            r'\b(?:salicylic|glycolic|lactic|mandelic|azelaic|kojic|arbutin)\b',
            r'\b(?:retinol|retinoid|tretinoin|adapalene|tazarotene)\b',
            r'\b(?:niacinamide|ascorbic|tocopherol|ceramide|peptide)\b',
            r'\b(?:benzoyl\s+peroxide|tea\s+tree|zinc\s+oxide)\b',
            
            # Skin conditions and concerns
            r'\b(?:acne|rosacea|eczema|dermatitis|psoriasis|melasma)\b',
            r'\b(?:hyperpigmentation|photoaging|wrinkles|fine\s+lines)\b',
            r'\b(?:blackheads|whiteheads|comedones|sebum|keratin)\b',
            
            # Medical conditions
            r'\b(?:hypercholesterolemia|dyslipidemia|atherosclerosis|coronary)\b',
            r'\b(?:inflammation|oxidation|antioxidant|biomarker)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    term = ' '.join(match)
                else:
                    term = match
                
                if len(term) > 2 and term not in medical_terms:
                    medical_terms.append(term)
        
                return medical_terms[:5]  # Return top 5 terms
    
    def _extract_core_claim(self, claim_text):
        """
        Extract core medical/cosmetic components from marketing-style claims
        """
        # Remove marketing fluff and extract key components
        claim_lower = claim_text.lower()
        
        # Remove common marketing phrases
        marketing_phrases = [
            r'\bstart with this\b',
            r'\bgentle\b',
            r'\bmild\b', 
            r'\bnew\b',
            r'\bimproved\b',
            r'\badvanced\b',
            r'\bprofessional\b',
            r'\bclinically tested\b',
            r'\bdermatologist recommended\b',
            r'\bfor best results\b',
            r'\bdaily use\b',
            r'\btwice daily\b'
        ]
        
        cleaned_claim = claim_lower
        for phrase in marketing_phrases:
            cleaned_claim = re.sub(phrase, '', cleaned_claim)
        
        # Extract key active ingredients and products
        key_components = []
        
        # Look for active ingredients
        ingredients = [
            'salicylic acid', 'glycolic acid', 'lactic acid', 'hyaluronic acid',
            'retinol', 'niacinamide', 'vitamin c', 'vitamin e', 'benzoyl peroxide',
            'azelaic acid', 'kojic acid', 'arbutin', 'ceramides', 'peptides'
        ]
        
        for ingredient in ingredients:
            if ingredient in cleaned_claim:
                key_components.append(ingredient)
        
        # Look for product types
        products = [
            'cleanser', 'serum', 'moisturizer', 'cream', 'lotion', 
            'toner', 'mask', 'exfoliant', 'sunscreen'
        ]
        
        for product in products:
            if product in cleaned_claim:
                key_components.append(product)
        
        # Combine into a cleaner claim
        if key_components:
            core_claim = ' '.join(key_components)
        else:
            # Fallback to cleaned version
            core_claim = ' '.join(cleaned_claim.split())
        
        return core_claim if core_claim.strip() else claim_text
    
    def _extract_key_concepts(self, text):
        """
        Extract key medical concepts and their relationships for query generation
        """
        text_lower = text.lower()
        
        # Key medical concepts to look for
        key_concepts = {
            'substance': None,  # The main intervention/substance
            'condition': None,  # The condition being addressed
            'effect': None,     # The primary effect/outcome
            'target': None      # The target of the effect (e.g., "blood pressure")
        }
        
        # Pattern matching for substance (intervention) - ENHANCED
        substance_patterns = [
            # Plant sterols/stanols specific patterns
            r'\b(plant\s+sterols?(?:\s+and\s+stanols?)?)\b',
            r'\b(plant\s+stanols?)\b',
            r'\b(phytosterols?(?:\s+and\s+phytostanols?)?)\b',
            r'\b(sterol\s+esters?)\b',
            r'\b(stanol\s+esters?)\b',
            
            # Fiber patterns
            r'\b(fiber|dietary fiber|soluble fiber|insoluble fiber|psyllium\s+husk?)\b',
            
            # Nutrients
            r'\b(vitamin\s+[a-zA-Z]\d?|niacin|vitamin\s+b3)\b',
            r'\b(magnesium|calcium|iron|zinc|omega[\s-]?\d+)\b',
            
            # General patterns
            r'\b([\w\s]+supplement(?:ation)?)\b',
            r'\b([\w\s]+(?:oil|acid|extract|compound))\b',
            r'\b([\w\s]+(?:therapy|treatment|intervention))\b'
        ]
        
        for pattern in substance_patterns:
            match = re.search(pattern, text_lower)
            if match:
                key_concepts['substance'] = match.group(1).strip()
                break
        
        # Pattern matching for conditions
        condition_patterns = [
            r'\b(constipation|diarrhea|bowel|intestinal|digestive)\b',
            r'\b([\w\s]+deficiency)\b',
            r'\b(hypertension|diabetes|obesity|cancer|[\w\s]+disease|[\w\s]+disorder)\b',
            r'\b([\w\s]+syndrome|[\w\s]+condition)\b',
            r'\b(cholesterol\s+absorption|lipid\s+absorption)\b'
        ]
        
        for pattern in condition_patterns:
            match = re.search(pattern, text_lower)
            if match:
                key_concepts['condition'] = match.group(1).strip()
                break
        
        # Pattern matching for effects - ENHANCED
        effect_patterns = [
            r'\b(block|blocks|blocking|inhibit|inhibits|inhibiting)\b',
            r'\b(helps|help|helps to|reduce|improve|increase|decrease|prevent|treat|regulate|support)(?:s|ing|ed)?\b',
            r'\b(retain|retains|retention|supplement(?:ing|ation)?|treat(?:ing|ment)?|prevent(?:ing|ion)?)\b',
            r'\b(lower|lowers|lowering|raise|raises|raising)\b'
        ]
        
        for pattern in effect_patterns:
            match = re.search(pattern, text_lower)
            if match:
                key_concepts['effect'] = match.group(1).strip()
                break
        
        # Pattern matching for targets - ENHANCED
        target_patterns = [
            r'(?:block|inhibit|reduce|lower|affect|impact)\s+(?:the\s+)?(\w+\s+absorption)',
            r'\b(cholesterol\s+absorption|lipid\s+absorption)\b',
            r'\b(ldl\s+cholesterol|hdl\s+cholesterol|total\s+cholesterol|ldl\s+and\s+total\s+cholesterol)\b',
            r'\b(triglycerides?|triglyceride\s+levels?)\b',
            r'\b(blood\s+pressure|heart\s+rhythm|muscle\s+function|cardiovascular\s+(?:events|risk))\b',
            r'\b(stool|bowel movement|transit time|intestines|intestinal transit|water|hydration)\b',
            r'\b([\w\s]+levels?|[\w\s]+function|[\w\s]+health)\b'
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, text_lower)
            if match:
                key_concepts['target'] = match.group(1).strip()
                break
        
        return key_concepts
    
    def _generate_search_queries_enhanced(self, health_claim, api_key=None):
        """
        Enhanced search query generation with GPT-powered intelligence and robust fallback hierarchy
        """
        # Enhanced approach with quality tracking
        all_queries = []
        query_sources = []
        
        # First: Try the focused approach with GPT integration
        focused_queries = self._generate_focused_search_queries(health_claim, api_key)
        
        if focused_queries:
            all_queries.extend(focused_queries)
            query_sources.extend(['focused'] * len(focused_queries))
            
            # If we got good quality queries from focused approach, prioritize them
            if len(focused_queries) >= 3:
                self._log_query_generation("Enhanced", len(focused_queries), "Using focused queries")
                return self._deduplicate_queries(all_queries)[:8]
        
        # Second: Priority drug name extraction for pharmaceutical claims  
        drug_names = self._extract_drug_names(health_claim)
        if drug_names:
            drug_queries = []
            
            for drug in drug_names[:2]:
                # Add exact drug name
                drug_queries.append(f'"{drug}"')
                query_sources.append('specific_drug')
                
                # Add drug with outcomes
                for outcome in ['efficacy', 'safety', 'cardiovascular outcomes'][:2]:
                    drug_queries.append(f'"{drug}" AND {outcome}')
                    query_sources.append('drug_outcome')
                
                # Add drug class context
                drug_class = self._get_drug_class(drug)
                if drug_class and drug_class != drug:
                    drug_queries.append(f'"{drug}" AND "{drug_class}"')
                    query_sources.append('drug_class')
            
            all_queries.extend(drug_queries)
            self._log_query_generation("Enhanced-Drugs", len(drug_queries), f"Found drugs: {', '.join(drug_names)}")
        
        # Third: Extract entities and build additional queries
        entities = self._extract_medical_entities(health_claim)
        entity_queries = []
        
        # Build queries from compound terms first (highest priority)
        if entities.get('compound_terms'):
            for term in entities['compound_terms'][:3]:
                entity_queries.append(f'"{term}"')
                query_sources.append('compound_term')
                
                # Add effect-based queries for known interventions
                if any(intervention in term.lower() for intervention in 
                      ['omega-3', 'fiber', 'psyllium', 'sterol', 'niacin', 'vitamin', 'mineral', 'extract']):
                    entity_queries.append(f'"{term}" AND effects')
                    entity_queries.append(f'"{term}" AND "clinical trial"')
                    query_sources.extend(['compound_effect', 'compound_trial'])
        
        # Add intervention-outcome combinations
        if entities.get('interventions') and entities.get('outcomes'):
            for intervention in entities['interventions'][:2]:
                for outcome in entities['outcomes'][:2]:
                    entity_queries.append(f'"{intervention}" AND "{outcome}"')
                    query_sources.append('intervention_outcome')
        
        # Add single intervention queries
        if entities.get('interventions'):
            for intervention in entities['interventions'][:2]:
                entity_queries.append(f'"{intervention}"')
                query_sources.append('intervention')
        
        # Third: Add medical condition queries
        if entities.get('medical_conditions'):
            for condition in entities['medical_conditions'][:2]:
                entity_queries.append(f'"{condition}"')
                query_sources.append('condition')
        
        # Validate entity queries
        validated_entity_queries = self._validate_search_queries(entity_queries, health_claim)
        
        # Combine all queries
        all_queries.extend(validated_entity_queries)
        
        # Fourth: Add triglyceride-specific queries if relevant
        if self._is_triglyceride_related(health_claim):
            trig_queries = self._generate_triglyceride_optimized_queries(health_claim)
            validated_trig = self._validate_search_queries(trig_queries, health_claim)
            all_queries.extend(validated_trig)
            query_sources.extend(['triglyceride'] * len(validated_trig))
        
        # Fifth: Emergency fallback with keyword extraction
        if len(all_queries) < 2:
            emergency_terms = self._extract_medical_terms_robust(health_claim)
            if emergency_terms:
                emergency_queries = [f'"{term}"' for term in emergency_terms[:3]]
                all_queries.extend(emergency_queries)
                query_sources.extend(['emergency'] * len(emergency_queries))
            else:
                # Ultimate fallback
                fallback_query = health_claim[:50].strip()
                if len(fallback_query) > 10:
                    all_queries.append(fallback_query)
                    query_sources.append('ultimate_fallback')
                else:
                    all_queries.append("health intervention clinical trial")
                    query_sources.append('generic')
        
        # Remove duplicates while preserving source order
        final_queries = self._deduplicate_queries(all_queries)
        
        # Log the final result
        self._log_query_generation("Enhanced", len(final_queries), 
                                 f"Sources: {', '.join(set(query_sources[:len(final_queries)]))}", 
                                 final_queries)
        
        return final_queries[:8]
    
    def _deduplicate_queries(self, queries):
        """
        Remove duplicate queries while preserving order and prioritizing better queries
        """
        seen = set()
        unique_queries = []
        
        for query in queries:
            if not query:
                continue
                
            # Normalize for comparison (remove extra spaces, case)
            normalized = ' '.join(query.lower().split())
            
            if normalized not in seen and len(normalized) > 3:
                unique_queries.append(query)
                seen.add(normalized)
        
        return unique_queries
    
    def _is_triglyceride_related(self, claim_text):
        """
        Check if claim is triglyceride-related for specialized query generation
        """
        triglyceride_terms = [
            'triglyceride', 'triglycerides', 'triacylglycerol', 'hypertriglyceridemia',
            'chylomicron', 'vldl', 'tg level', 'lipemia', 'milky plasma'
        ]
        
        claim_lower = claim_text.lower()
        return any(term in claim_lower for term in triglyceride_terms)
    
    def test_query_generation(self, test_claim, show_fallbacks=False):
        """
        Test and demonstrate the enhanced query generation with different methods
        
        Args:
            test_claim: Health claim to test
            show_fallbacks: Whether to show all fallback methods
            
        Returns:
            Dict with results from different generation methods
        """
        results = {}
        
        print(f"\n🔍 Testing Query Generation for: '{test_claim}'\n")
        
        # Test GPT-powered generation
        if self.api_key:
            print("1️⃣ Testing GPT-powered generation...")
            gpt_queries = self.generate_pubmed_queries_with_gpt(test_claim, self.api_key)
            results['gpt'] = gpt_queries
            print(f"   Generated {len(gpt_queries)} queries: {gpt_queries}\n")
        else:
            print("1️⃣ GPT generation skipped (no API key)\n")
            results['gpt'] = []
        
        if show_fallbacks:
            # Test smart medical query generation
            print("2️⃣ Testing smart medical generation...")
            smart_queries = self._generate_smart_medical_queries(test_claim)
            results['smart'] = smart_queries
            print(f"   Generated {len(smart_queries)} queries: {smart_queries}\n")
            
            # Test basic intervention-outcome extraction
            print("3️⃣ Testing basic extraction...")
            components = self._extract_intervention_and_outcome(test_claim)
            basic_queries = []
            if components['interventions'] and components['outcomes']:
                for intervention in components['interventions'][:2]:
                    for outcome in components['outcomes'][:2]:
                        basic_queries.append(f'"{intervention}" AND "{outcome}"')
            results['basic'] = basic_queries
            print(f"   Generated {len(basic_queries)} queries: {basic_queries}\n")
            
            # Test medical term extraction
            print("4️⃣ Testing medical term extraction...")
            medical_terms = self._extract_medical_terms_robust(test_claim)
            term_queries = [f'"{term}"' for term in medical_terms[:3]]
            results['terms'] = term_queries
            print(f"   Extracted terms: {medical_terms}")
            print(f"   Generated {len(term_queries)} queries: {term_queries}\n")
        
        # Test the full enhanced pipeline
        print("🚀 Testing full enhanced pipeline...")
        enhanced_queries = self._generate_search_queries_enhanced(test_claim, self.api_key)
        results['enhanced'] = enhanced_queries
        print(f"   Final result: {len(enhanced_queries)} queries: {enhanced_queries}\n")
        
        return results
    
    def debug_query_failure(self, test_claim):
        """
        Debug why query generation is failing for a specific claim
        """
        print(f"\n🔍 DEBUGGING QUERY FAILURE for: '{test_claim}'\n")
        
        # Test each step of the pipeline
        print("1️⃣ Testing drug name extraction...")
        drug_names = self._extract_drug_names(test_claim)
        print(f"   Drug names: {drug_names}")
        for drug in drug_names:
            drug_class = self._get_drug_class(drug)
            print(f"   {drug} -> class: {drug_class}")
        
        print("\n2️⃣ Testing medical term extraction...")
        medical_terms = self._extract_medical_terms_robust(test_claim)
        print(f"   Extracted terms: {medical_terms}")
        
        print("\n3️⃣ Testing intervention-outcome extraction...")
        components = self._extract_intervention_and_outcome(test_claim)
        print(f"   Interventions: {components.get('interventions', [])}")
        print(f"   Outcomes: {components.get('outcomes', [])}")
        
        print("\n4️⃣ Testing smart medical query generation...")
        smart_queries = self._generate_smart_medical_queries(test_claim)
        print(f"   Smart queries: {smart_queries}")
        
        print("\n5️⃣ Testing validation on sample queries...")
        test_queries = [
            f'"{test_claim[:30]}"',
            '"evacetrapib"',
            '"ezetimibe" AND statins',
            '"salicylic acid" AND acne',
            '"cleanser" AND "skin care"',
            'cardiovascular outcomes'
        ]
        
        for query in test_queries:
            is_valid = len(self._validate_search_queries([query], test_claim)) > 0
            print(f"   Query: {query} -> {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        print("\n6️⃣ Testing full pipeline with logging...")
        original_logging = getattr(self, '_enable_logging', False)
        self._enable_logging = True
        
        final_queries = self._generate_search_queries_enhanced(test_claim, self.api_key)
        
        self._enable_logging = original_logging
        print(f"   Final queries: {final_queries}")
        
        return {
            'drug_names': drug_names,
            'medical_terms': medical_terms,
            'components': components, 
            'smart_queries': smart_queries,
            'final_queries': final_queries
        }
    
    def _generate_triglyceride_optimized_queries(self, claim_text):
        """Generate highly specific queries for triglyceride claims"""
        queries = []
        
        # Detect if it's a triglyceride claim
        trig_indicators = ['triglyceride', 'tg', 'hypertriglyceridemia', 'lipid', 'chylomicron']
        is_trig_claim = any(ind in claim_text.lower() for ind in trig_indicators)
        
        if is_trig_claim:
            # Extract key components
            expert = TriglycerideExpert()
            value_data = expert.validate_triglyceride_value(claim_text)
            
            # Base queries for triglycerides
            queries.extend([
                '"hypertriglyceridemia" AND treatment',
                '"triglycerides" AND ("systematic review"[Publication Type] OR "meta-analysis"[Publication Type])',
                '"severe hypertriglyceridemia" AND management'
            ])
            
            # If specific intervention mentioned
            interventions = expert.validate_intervention_claim(claim_text)
            for intervention in interventions:
                if 'intervention' in intervention:
                    queries.append(f'"{intervention["intervention"]}" AND triglycerides AND ("clinical trial"[Publication Type])')
            
            # If value range mentioned
            if value_data and value_data['value'] > 500:
                queries.append('"triglycerides" AND "pancreatitis" AND prevention')
            
            # Add guideline-specific searches
            queries.extend([
                '"Endocrine Society" AND "hypertriglyceridemia" AND guideline',
                '"American Heart Association" AND triglycerides AND management',
                '"National Lipid Association" AND hypertriglyceridemia'
            ])
        
        # Continue with general query generation for non-triglyceride claims
        else:
            queries = self._generate_search_queries_enhanced(claim_text, self.api_key)
        
        return queries[:8]  # Limit queries
    
    def test_pharmaceutical_claims(self):
        """
        Test query generation for pharmaceutical claims specifically
        """
        test_claims = [
            "The drug evacetrapib, which lowered LDL levels by 37%, had no effect on heart health",
            "Adding ezetimibe to a statin does not improve outcomes, suggesting that only statins show significant benefits",
            "Semaglutide showed superior cardiovascular outcomes compared to placebo in diabetic patients",
            "Atorvastatin 20mg daily reduced LDL cholesterol by an average of 43% in clinical trials"
        ]
        
        print("🧪 TESTING PHARMACEUTICAL QUERY GENERATION\n")
        
        for i, claim in enumerate(test_claims, 1):
            print(f"Test {i}: {claim[:60]}...")
            
            # Extract drug names
            drugs = self._extract_drug_names(claim)
            print(f"   🔍 Drugs found: {drugs}")
            
            # Generate queries
            queries = self._generate_focused_search_queries(claim, self.api_key)
            print(f"   📝 Generated queries:")
            for j, query in enumerate(queries[:3], 1):
                print(f"      {j}. {query}")
            
            print()
    
    def score_article_quality(self, article):
        """
        Score article quality based on study type, journal impact, and recency
        """
        score = 0
        
        # 1. Publication type scoring (evidence hierarchy)
        pub_types = [pt.lower() for pt in article.get('publication_types', [])]
        
        evidence_hierarchy = {
            'systematic review': 100,
            'meta-analysis': 95,
            'practice guideline': 90,
            'randomized controlled trial': 80,
            'clinical trial': 70,
            'cohort study': 60,
            'case-control study': 50,
            'observational study': 40,
            'case report': 30,
            'editorial': 20,
            'letter': 10,
            'comment': 5
        }
        
        # Find highest scoring publication type
        for pub_type, points in evidence_hierarchy.items():
            if any(pub_type in pt for pt in pub_types):
                score = max(score, points)
                break
        
        # 2. Journal quality indicators
        journal = article.get('journal', '').lower()
        high_impact_journals = [
            'new england journal', 'lancet', 'jama', 'bmj', 'nature medicine',
            'annals of internal medicine', 'circulation', 'gastroenterology',
            'journal of clinical oncology', 'diabetes care', 'chest'
        ]
        
        if any(hij in journal for hij in high_impact_journals):
            score += 20
        
        # 3. Recency bonus (more recent = better for clinical guidelines)
        try:
            year = int(article.get('year', 0))
            current_year = datetime.now().year
            if year >= current_year - 2:
                score += 15  # Very recent
            elif year >= current_year - 5:
                score += 10  # Recent
            elif year >= current_year - 10:
                score += 5   # Somewhat recent
        except:
            pass
        
        # 4. Sample size indicator (if mentioned in abstract)
        abstract = article.get('abstract', '').lower()
        large_sample_indicators = [r'n\s*=\s*\d{3,}', r'n\s*>\s*\d{3,}', r'\d{3,}\s*patients', r'\d{3,}\s*participants']
        if any(re.search(pattern, abstract) for pattern in large_sample_indicators):
            score += 10
        
        return score
    
    def search(self, query, max_results=30):
        """
        Search PubMed for articles matching the query with better error handling
        
        Args:
            query: Search terms
            max_results: Maximum number of results
            
        Returns:
            List of PubMed IDs
        """
        # Check cache first
        cache_key = f"{query}:{max_results}"
        if cache_key in self.cache:
            st.info(f"Using cached results for query: {query[:50]}...")
            return self.cache[cache_key]
        
        # Log the actual query being sent to PubMed
        st.info(f"PubMed search query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance"
        }
        
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(self.PUBMED_SEARCH_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for errors in the response
                if "esearchresult" in data:
                    if "ERROR" in data["esearchresult"]:
                        st.error(f"PubMed API error: {data['esearchresult']['ERROR']}")
                        return []
                    
                    id_list = data["esearchresult"].get("idlist", [])
                    count = data["esearchresult"].get("count", "0")
                    
                    # Report how many articles were found
                    st.info(f"Found {len(id_list)} articles (Total available: {count})")
                    
                    # Store in cache
                    self.cache[cache_key] = id_list
                    
                    return id_list
                else:
                    st.error("Unexpected response format from PubMed")
                    return []
            
            except requests.exceptions.RequestException as e:
                st.error(f"Network error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    st.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
            except Exception as e:
                st.error(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    st.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        
        return []
    
    def fetch_articles(self, pmids):
        """
        Fetch details of articles from PubMed given their IDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article details
        """
        if not pmids:
            return []
        
        # Process in larger batches for efficiency
        BATCH_SIZE = 10  # Increased from 5 to 10 for more efficiency
        all_articles = []
        
        # Split the PMIDs into smaller batches
        pmid_batches = [pmids[i:i + BATCH_SIZE] for i in range(0, len(pmids), BATCH_SIZE)]
        
        for batch_num, pmid_batch in enumerate(pmid_batches):
            st.info(f"Fetching details for article batch {batch_num + 1}/{len(pmid_batches)} (IDs: {', '.join(pmid_batch)})")
            
            # Wait between batches to respect rate limits (skip first batch)
            if batch_num > 0:
                wait_time = self.SEARCH_DELAY  # Shorter delay with API key
                st.info(f"Waiting {wait_time} seconds between article fetches...")
                time.sleep(wait_time)
            
            params = {
                "db": "pubmed",
                "id": ",".join(pmid_batch),
                "retmode": "xml"
            }
            
            if self.email:
                params["email"] = self.email
            if self.api_key:
                params["api_key"] = self.api_key
            
            articles_fetched = False
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = requests.get(self.PUBMED_FETCH_URL, params=params)
                    response.raise_for_status()
                    
                    # Parse XML response
                    root = ET.fromstring(response.content)
                    batch_articles = []
                    
                    for article_elem in root.findall(".//PubmedArticle"):
                        try:
                            # Extract PMID
                            pmid_elem = article_elem.find(".//PMID")
                            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
                            
                            # Extract title
                            title_elem = article_elem.find(".//ArticleTitle")
                            title = title_elem.text if title_elem is not None else "No title available"
                            
                            # Extract abstract
                            abstract_texts = article_elem.findall(".//AbstractText")
                            abstract = " ".join([text.text for text in abstract_texts if text.text]) if abstract_texts else "No abstract available"
                            
                            # Extract authors
                            author_elems = article_elem.findall(".//Author")
                            authors = []
                            for author_elem in author_elems:
                                lastname = author_elem.find("LastName")
                                forename = author_elem.find("ForeName")
                                if lastname is not None and forename is not None:
                                    authors.append(f"{lastname.text} {forename.text}")
                                elif lastname is not None:
                                    authors.append(lastname.text)
                            
                            authors_str = ", ".join(authors) if authors else "Unknown authors"
                            
                            # Extract journal and date
                            journal_elem = article_elem.find(".//Journal/Title")
                            journal = journal_elem.text if journal_elem is not None else "Unknown journal"
                            
                            year_elem = article_elem.find(".//PubDate/Year")
                            year = year_elem.text if year_elem is not None else "Unknown year"
                            
                            # Check for open access status - keep the check but don't filter by it
                            is_open_access = False
                            # Look for OA indicators in the XML
                            oa_elements = article_elem.findall(".//PublicationStatus")
                            for oa_elem in oa_elements:
                                if oa_elem.text and "open access" in oa_elem.text.lower():
                                    is_open_access = True
                                    break
                            
                            # Double-check using attributes that indicate open access
                            publication_type_elems = article_elem.findall(".//PublicationType")
                            publication_types = []
                            for elem in publication_type_elems:
                                if elem.text:
                                    publication_types.append(elem.text)
                                    if "open access" in elem.text.lower():
                                        is_open_access = True
                            
                            # Extract DOI if available
                            doi = None
                            doi_elems = article_elem.findall(".//ArticleId[@IdType='doi']")
                            if doi_elems and doi_elems[0].text:
                                doi = doi_elems[0].text
                            
                            # Generate PubMed URL
                            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "Unknown" else None
                            
                            # Generate DOI URL if DOI exists
                            doi_url = f"https://doi.org/{doi}" if doi else None
                            
                            # Create article object
                            article = {
                                "pmid": pmid,
                                "title": title,
                                "abstract": abstract,
                                "authors": authors_str,
                                "journal": journal,
                                "year": year,
                                "pubmed_url": pubmed_url,
                                "doi": doi,
                                "doi_url": doi_url,
                                "is_open_access": is_open_access,
                                "citation": f"{authors_str}. {title}. {journal}, {year}.",
                                "publication_types": publication_types
                            }
                            
                            # Calculate quality score
                            article['quality_score'] = self.score_article_quality(article)
                            
                            batch_articles.append(article)
                        
                        except Exception as e:
                            st.error(f"Error parsing article: {str(e)}")
                    
                    all_articles.extend(batch_articles)
                    articles_fetched = True
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    st.error(f"Fetch attempt {attempt + 1} failed for batch {batch_num + 1}: {str(e)}")
                    if attempt < self.MAX_RETRIES - 1:
                        wait_time = self.RETRY_DELAY * (2 ** attempt)
                        st.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
            
            # If all retries failed for this batch, continue to the next batch
            if not articles_fetched:
                st.warning(f"Could not fetch articles for batch {batch_num + 1} after {self.MAX_RETRIES} attempts.")
                
        # Sort articles by quality score
        all_articles.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        # Report open access vs. non-open access counts
        open_count = sum(1 for article in all_articles if article.get("is_open_access", False))
        st.info(f"Retrieved {len(all_articles)} articles ({open_count} are open access)")
        
        return all_articles
    
    def _expand_term_with_synonyms(self, term):
        """
        Expand a medical term with its common synonyms for better search coverage
        
        Args:
            term: Medical term to expand
            
        Returns:
            List of the original term and its synonyms
        """
        # Convert term to lowercase for dictionary lookup
        term_lower = term.lower()
        
        # Check if the term exists in our synonym dictionary
        if term_lower in self.medical_synonyms:
            # Return original term and synonyms
            return [term] + self.medical_synonyms[term_lower]
        
        # Check if term is a synonym of another term (reverse lookup)
        for main_term, synonyms in self.medical_synonyms.items():
            if term_lower in [s.lower() for s in synonyms]:
                return [term, main_term] + [s for s in synonyms if s.lower() != term_lower]
        
        # No synonyms found, return just the original term
        return [term]
    
    def find_relevant_articles_simple(self, health_claim, max_results=10):
        """
        Simplified article search for YouTube claims - fewer queries, broader terms
        """
        # Extract basic medical terms
        medical_terms = re.findall(r'\b(health|vitamin|diet|exercise|treatment|disease|cancer|heart|brain|supplement|medicine|therapy|cure|diabetes|blood|pressure|weight|immune|infection|pain|sleep|stress|anxiety|depression)\b', health_claim.lower())
        
        if not medical_terms:
            # If no medical terms found, try to extract any significant words
            words = re.findall(r'\b[a-z]{4,}\b', health_claim.lower())
            stopwords = {'this', 'that', 'with', 'from', 'have', 'what', 'when', 'where', 'which', 'your', 'their', 'these', 'those'}
            medical_terms = [w for w in words if w not in stopwords][:3]
        
        if not medical_terms:
            st.warning(f"No searchable medical terms found in claim: {health_claim[:50]}...")
            return []
        
        # Create simple queries
        queries = []
        
        # Query 1: Simple OR of main terms
        if medical_terms:
            simple_query = " OR ".join(medical_terms[:3])
            queries.append(simple_query)
        
        # Query 2: Look for high-quality evidence on the main term
        if medical_terms:
            quality_query = f"{medical_terms[0]} AND (systematic review OR meta-analysis)"
            queries.append(quality_query)
        
        # Search with simplified queries
        all_pmids = []
        for i, query in enumerate(queries):
            st.info(f"Simple search {i+1}: {query}")
            pmids = self.search(query, max_results=max_results)
            if pmids:
                all_pmids.extend(pmids[:5])  # Take only top 5 from each query
        
        # Remove duplicates
        unique_pmids = []
        for pmid in all_pmids:
            if pmid not in unique_pmids:
                unique_pmids.append(pmid)
        
        unique_pmids = unique_pmids[:max_results]
        
        if unique_pmids:
            articles = self.fetch_articles(unique_pmids)
            return articles
        
        return []
    
    def find_relevant_articles(self, health_claim, max_results=20):
        """
        Updated to use the improved query generation
        """
        # Check if it's a triglyceride-related claim
        trig_indicators = ['triglyceride', 'tg', 'hypertriglyceridemia', 'lipid', 'chylomicron']
        is_trig_claim = any(ind in health_claim.lower() for ind in trig_indicators)
        
        # For triglyceride claims, check if there's a specific intervention mentioned
        if is_trig_claim:
            components = self._extract_intervention_and_outcome(health_claim)
            # If specific interventions are mentioned, use general search instead of triglyceride-specific
            if components['interventions']:
                search_queries = self._generate_search_queries_enhanced(health_claim, self.api_key)
            else:
                # Only use triglyceride-specific queries if no intervention is mentioned
                search_queries = self._generate_triglyceride_optimized_queries(health_claim)
        else:
            search_queries = self._generate_search_queries_enhanced(health_claim, self.api_key)
        
        if not search_queries:
            # Final fallback
            search_queries = [health_claim[:50]]
        
        # Log the queries for debugging
        st.success(f"Generated {len(search_queries)} optimized PubMed search queries")
        with st.expander("View search queries"):
            for i, query in enumerate(search_queries):
                st.write(f"{i+1}. {query}")
        
        all_pmids = []
        queries_with_results = 0
        
        # Execute searches
        for i, query in enumerate(search_queries):
            st.info(f"Executing search query {i+1}/{len(search_queries)}: {query[:100]}...")
            pmids = self.search(query, max_results=max_results)
            
            if pmids:
                queries_with_results += 1
                all_pmids.extend(pmids)
                st.success(f"Query {i+1} found {len(pmids)} articles")
            else:
                st.warning(f"Query {i+1} found no results")
            
            # Add delay between searches
            if i < len(search_queries) - 1:
                time.sleep(self.SEARCH_DELAY)
        
        # Remove duplicates while preserving order
        unique_pmids = []
        seen = set()
        for pmid in all_pmids:
            if pmid not in seen:
                unique_pmids.append(pmid)
                seen.add(pmid)
        
        # Limit to max_results
        unique_pmids = unique_pmids[:max_results]
        
        # Fetch articles
        if unique_pmids:
            st.success(f"Found {len(unique_pmids)} unique articles across all searches")
            articles = self.fetch_articles(unique_pmids)
            return articles
        
        return []
    
    def _clean_and_validate_query(self, query):
        """
        Clean up malformed queries and ensure they're valid for PubMed
        """
        if not query or len(query.strip()) < 3:
            return None
        
        # Remove incomplete sentences (ending with partial words)
        if query.endswith(('reducing', 'cau', 'the', 'and', 'or', 'of', 'in', 'to', 'for')):
            # Try to extract the main concept before the incomplete ending
            words = query.split()
            if len(words) > 3:
                # Take the first meaningful part
                main_concept = ' '.join(words[:3])
                if any(term in main_concept.lower() for term in ['fiber', 'stool', 'psyllium', 'constipation']):
                    return f'"{main_concept.strip()}"'
        
        # Remove quotes from partial sentences and convert to keywords
        cleaned = query.replace('"', '').strip()
        
        # If it looks like a sentence fragment, extract key medical terms
        if any(word in cleaned.lower() for word in ['helps', 'can', 'will', 'may', 'is', 'are']):
            # Extract key terms for fiber-related claims
            fiber_terms = []
            text_lower = cleaned.lower()
            
            # Look for fiber-related keywords
            if 'psyllium' in text_lower:
                fiber_terms.append('psyllium')
            if any(term in text_lower for term in ['fiber', 'bulk']):
                fiber_terms.append('dietary fiber')
            if any(term in text_lower for term in ['stool', 'bowel']):
                fiber_terms.append('stool consistency')
            if any(term in text_lower for term in ['water', 'absorb', 'retain']):
                fiber_terms.append('water absorption')
            if any(term in text_lower for term in ['transit', 'movement', 'move']):
                fiber_terms.append('intestinal transit')
            if 'constipation' in text_lower:
                fiber_terms.append('constipation')
            if any(term in text_lower for term in ['colon', 'intestin']):
                fiber_terms.append('colonic function')
            
            if fiber_terms:
                # Combine the most relevant terms
                if len(fiber_terms) >= 2:
                    return f'"{fiber_terms[0]}" AND "{fiber_terms[1]}"'
                else:
                    return f'"{fiber_terms[0]}"'
        
        # If query is too long (likely a sentence), extract key terms
        if len(cleaned) > 50:
            words = cleaned.split()
            key_words = [w for w in words if len(w) > 3 and w.lower() not in 
                        ['helps', 'this', 'that', 'with', 'from', 'have', 'what', 
                        'when', 'where', 'which', 'your', 'their', 'these', 'those',
                        'can', 'will', 'may', 'should', 'could', 'would']]
            if key_words:
                # Take first 2-3 key medical terms
                return ' AND '.join([f'"{word}"' for word in key_words[:3]])
        
        # Return cleaned query with quotes if it's reasonable length
        if 5 <= len(cleaned) <= 50:
            return f'"{cleaned}"'
        
        return None

def format_articles_for_analysis(articles, quality_threshold=50):
    """Format articles with quality indicators"""
    if not articles:
        return "No PubMed articles found."
    
    formatted = []
    for i, article in enumerate(articles):
        if article.get('quality_score', 0) >= quality_threshold:
            section = f"Article {i+1} (Quality Score: {article.get('quality_score', 'N/A')}):\n"
            section += f"Title: {article['title']}\n"
            section += f"Type: {', '.join(article.get('publication_types', ['Unknown']))}\n"
            section += f"Authors: {article['authors']}\n"
            section += f"Journal: {article['journal']}, {article['year']}\n"
            section += f"PMID: {article['pmid']}\n"
            
            if article.get('doi'):
                section += f"DOI: {article['doi']}\n"
                
            section += f"Abstract: {article['abstract']}\n\n"
            formatted.append(section)
    
    return "\n".join(formatted)

def extract_key_guideline_points(guideline_text, claim, max_length=300):
    """
    Extract the most relevant sentences from a guideline for a specific claim
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', guideline_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return guideline_text[:max_length]
    
    # Keywords that indicate important guideline content
    importance_keywords = [
        'recommend', 'should', 'must', 'guideline', 'evidence', 'consensus',
        'first-line', 'treatment', 'diagnosis', 'management', 'indicated',
        'contraindicated', 'grade', 'level', 'class'
    ]
    
    # Extract key terms from the claim
    claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
    
    # Score each sentence
    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        
        # Score based on claim relevance
        for word in claim_words:
            if word in sentence_lower:
                score += 2
        
        # Score based on guideline importance
        for keyword in importance_keywords:
            if keyword in sentence_lower:
                score += 3
        
        if score > 0:
            scored_sentences.append((score, sentence))
    
    # Sort by score
    scored_sentences.sort(reverse=True)
    
    # Combine top sentences up to max_length
    result = []
    current_length = 0
    for score, sentence in scored_sentences:
        if current_length + len(sentence) < max_length:
            result.append(sentence)
            current_length += len(sentence)
        else:
            break
    
    return '. '.join(result) if result else guideline_text[:max_length]

def format_guidelines_for_citation(guidelines, claim):
    """
    Format guidelines in a way that makes them easy for the AI to cite
    """
    formatted_guidelines = []
    
    for i, guideline in enumerate(guidelines[:5]):  # Limit to top 5
        # Get the guideline content
        content = guideline.get('content', '')
        if not content and 'file_path' in guideline and os.path.exists(guideline['file_path']):
            try:
                with open(guideline['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                content = guideline.get('content_preview', '')
        
        # Extract key points relevant to the claim
        key_points = extract_key_guideline_points(content, claim) if content else "No content available"
        
        formatted_guide = {
            'citation_id': f'[G{i+1}]',
            'full_citation': f"{guideline['society']} ({guideline['year']}) - {guideline['category']}",
            'key_points': key_points,
            'quality_score': guideline.get('quality_score', 0),
            'relevance_score': guideline.get('relevance_score', 0)
        }
        
        formatted_guidelines.append(formatted_guide)
    
    return formatted_guidelines

def format_pubmed_for_citation(articles, claim):
    """
    Format PubMed articles in a way that makes them easy for the AI to cite
    """
    formatted_articles = []
    
    for i, article in enumerate(articles[:10]):  # Limit to top 10
        # Extract the most relevant finding from the abstract
        abstract = article.get('abstract', '')
        
        # Look for conclusion/result sentences
        conclusion_patterns = [
            r'(?:in conclusion|we found|results show|demonstrated that|our findings|the study found)[^.]+\.',
            r'(?:significantly|significant)\s+(?:reduced|increased|improved|associated)[^.]+\.',
            r'(?:effective|efficacy|benefit|risk)[^.]+\.'
        ]
        
        key_finding = ""
        for pattern in conclusion_patterns:
            match = re.search(pattern, abstract, re.IGNORECASE)
            if match:
                key_finding = match.group(0)
                break
        
        if not key_finding and abstract:
            # Just take the first substantial sentence
            sentences = re.split(r'[.!?]+', abstract)
            key_finding = next((s for s in sentences if len(s.strip()) > 50), abstract[:150])
        
        formatted_article = {
            'citation_id': f'[P{i+1}]',
            'pmid': article['pmid'],
            'title': article['title'],
            'key_finding': key_finding.strip() if key_finding else "See full abstract",
            'study_type': article.get('publication_types', ['Unknown'])[0],
            'quality_score': article.get('quality_score', 0)
        }
        
        formatted_articles.append(formatted_article)
    
    return formatted_articles

def format_pubmed_evidence_with_forced_citations(articles):
    """
    Format PubMed articles in a way that makes citations mandatory and clearer
    """
    if not articles:
        return "No PubMed articles found."
    
    formatted_sections = []
    
    # Group articles by quality
    high_quality = [a for a in articles if a.get('quality_score', 0) >= 80]
    medium_quality = [a for a in articles if 50 <= a.get('quality_score', 0) < 80]
    
    # Format high-quality evidence
    if high_quality:
        section = "HIGH-QUALITY EVIDENCE TO CITE:\n\n"
        for i, article in enumerate(high_quality[:5]):
            pub_type = article.get('publication_types', ['study'])[0].lower()
            
            section += f"MUST CITE #{i+1}: "
            section += f'"{article["title"]}" '
            section += f"(PMID: {article['pmid']})\n"
            section += f"TYPE: {pub_type}\n"
            section += f"KEY FINDING: {article['abstract'][:150]}...\n"
            section += f"EXAMPLE CITATION: 'A {pub_type} (PMID: {article['pmid']}) demonstrated that...'\n\n"
        
        formatted_sections.append(section)
    
    # Format medium-quality evidence
    if medium_quality:
        section = "MEDIUM-QUALITY EVIDENCE TO CITE:\n\n"
        for i, article in enumerate(medium_quality[:3]):
            pub_type = article.get('publication_types', ['study'])[0].lower()
            
            section += f"CITE IF RELEVANT #{i+1}: "
            section += f'"{article["title"]}" '
            section += f"(PMID: {article['pmid']})\n"
            section += f"TYPE: {pub_type}\n"
            section += f"KEY FINDING: {article['abstract'][:100]}...\n\n"
        
        formatted_sections.append(section)
    
    return "\n".join(formatted_sections)

def check_and_add_missing_citations(analysis, articles):
    """
    Post-process the analysis to ensure PubMed citations are included
    """
    if not articles:
        return analysis
    
    # Check if any PMIDs are cited in the analysis
    cited_pmids = re.findall(r'PMID:\s*(\d+)', analysis)
    
    # If no citations found, add a citation section
    if not cited_pmids and len(articles) > 0:
        # Find the evidence section or add one
        citation_section = "\n\n**Key Supporting Evidence (Added for transparency):**\n"
        
        # Add top 3 articles as citations
        for i, article in enumerate(articles[:3]):
            if article.get('quality_score', 0) >= 50:
                pub_type = article.get('publication_types', ['Study'])[0]
                citation_section += f"\n- {pub_type} (PMID: {article['pmid']}): "
                citation_section += f"{article['title'][:80]}..."
                if article.get('abstract'):
                    # Extract a key finding from the abstract
                    abstract_words = article['abstract'].split()[:30]
                    citation_section += f" Found that {' '.join(abstract_words)}..."
        
        # Insert before the credibility score or at the end
        if "**Credibility Score:**" in analysis:
            parts = analysis.split("**Credibility Score:**")
            analysis = parts[0] + citation_section + "\n**Credibility Score:**" + parts[1]
        else:
            analysis += citation_section
    
    return analysis

def enhance_analysis_with_links(analysis, article_data):
    """Add clickable links to PubMed and DOI references in the analysis"""
    if not article_data:
        return analysis
    
    # Create a dictionary for easy lookup
    article_dict = {article["pmid"]: article for article in article_data}
    
    # Add links to PubMed IDs
    for article in article_data:
        pmid = article["pmid"]
        if article.get('pubmed_url'):
            # Replace PMID references with links
            pattern = f"PMID: ?{pmid}"
            replacement = f"[PMID: {pmid}]({article['pubmed_url']})"
            analysis = re.sub(pattern, replacement, analysis)
            
            # Also look for the PMID in parentheses
            pattern = f"\\(PMID: ?{pmid}\\)"
            replacement = f"([PMID: {pmid}]({article['pubmed_url']}))"
            analysis = re.sub(pattern, replacement, analysis)
    
    # Add links to DOIs if available
    for article in article_data:
        if article.get('doi') and article.get('doi_url'):
            # Replace DOI references with links
            pattern = f"DOI: ?{re.escape(article['doi'])}"
            replacement = f"[DOI: {article['doi']}]({article['doi_url']})"
            analysis = re.sub(pattern, replacement, analysis)
            
            # Also look for the DOI in parentheses
            pattern = f"\\(DOI: ?{re.escape(article['doi'])}\\)"
            replacement = f"([DOI: {article['doi']}]({article['doi_url']}))"
            analysis = re.sub(pattern, replacement, analysis)
    
    return analysis

def detect_query_intent(text, api_key):
    """Determine if the input is a claim to verify or a question to answer"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    intent_detector_prompt = """
    Classify the following text as either:
    1. "CLAIM" - A statement about health that needs fact-checking
    2. "QUESTION" - A question seeking health information
    3. "MIXED" - Contains both claims and questions
    
    Return only one word: CLAIM, QUESTION, or MIXED
    """
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": intent_detector_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            intent = data["choices"][0]["message"]["content"].strip().upper()
            if intent in ["CLAIM", "QUESTION", "MIXED"]:
                return intent
        return "CLAIM"  # Default to claim
    except:
        return "CLAIM"  # Default to claim on error

def extract_main_medical_topic(text):
    """Extract the main medical topic from a question"""
    # Simple extraction based on common patterns
    text_lower = text.lower()
    
    # Common medical topics
    medical_topics = [
        'diabetes', 'hypertension', 'cancer', 'heart disease', 'stroke',
        'arthritis', 'depression', 'anxiety', 'asthma', 'allergy',
        'vitamin', 'supplement', 'medication', 'exercise', 'diet',
        'weight loss', 'pregnancy', 'vaccination', 'infection'
    ]
    
    for topic in medical_topics:
        if topic in text_lower:
            return topic
    
    # Extract longest medical term
    medical_terms = extract_medical_terms(text)
    if medical_terms:
        return max(medical_terms, key=len)
    
    return None

def generate_qa_search_queries(question):
    """Generate search queries optimized for Q&A rather than claim verification"""
    # Extract question type (what, how, why, when, etc.)
    question_words = ['what', 'how', 'why', 'when', 'which', 'who', 'can', 'does', 'is']
    
    # Extract key medical concepts
    medical_entities = extract_medical_entities_for_metadata(question)
    
    queries = []
    
    # Strategy 1: Direct question search
    queries.append(question)
    
    # Strategy 2: Concept-based search
    if medical_entities['conditions']:
        for condition in medical_entities['conditions'][:2]:
            queries.append(f"{condition} symptoms diagnosis treatment")
    
    # Strategy 3: Add "patient education" or "overview" for better results
    main_topic = extract_main_medical_topic(question)
    if main_topic:
        queries.append(f"{main_topic} patient education")
        queries.append(f"{main_topic} clinical overview")
    
    return queries[:5]

def check_medical_urgency(question):
    """Detect if question indicates urgent medical need"""
    urgent_patterns = [
        r'emergency|urgent|severe pain|chest pain|can\'t breathe|bleeding heavily',
        r'suicidal|want to die|kill myself',
        r'overdose|poisoning|took too much',
        r'heart attack|stroke symptoms'
    ]
    
    for pattern in urgent_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return True, "This sounds like a medical emergency. Please call emergency services (911) or visit the nearest emergency room immediately."
    
    return False, None

def answer_health_question(question, api_key, model="gpt-4o", use_guidelines=True, use_pubmed=True, ncbi_api_key=None, ncbi_email=None):
    """Answer health questions using guidelines and PubMed evidence"""
    import time
    start_time = time.time()
    
    # Search for relevant guidelines
    relevant_guidelines = []
    if use_guidelines:
        with st.spinner("Searching medical society guidelines..."):
            relevant_guidelines = get_relevant_guidelines(question, max_guidelines=5)
            if relevant_guidelines:
                st.success(f"Found {len(relevant_guidelines)} relevant guidelines")
    
    # Search PubMed for evidence
    articles = []
    if use_pubmed:
        with st.spinner("Searching medical literature..."):
            pubmed_searcher = PubMedSearcher(email=ncbi_email, api_key=ncbi_api_key)
            # Use Q&A optimized queries
            search_queries = generate_qa_search_queries(question)
            
            all_pmids = []
            for query in search_queries:
                pmids = pubmed_searcher.search(query, max_results=10)
                all_pmids.extend(pmids)
            
            # Remove duplicates
            unique_pmids = list(set(all_pmids))[:20]
            
            if unique_pmids:
                articles = pubmed_searcher.fetch_articles(unique_pmids)
                articles.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
                st.success(f"Found {len(articles)} relevant articles")
    
    # Format evidence
    citation_guide = "AVAILABLE EVIDENCE:\n\n"
    
    if relevant_guidelines:
        citation_guide += "MEDICAL SOCIETY GUIDELINES:\n"
        formatted_guidelines = format_guidelines_for_citation(relevant_guidelines, question)
        for fg in formatted_guidelines:
            citation_guide += f"{fg['citation_id']} = {fg['full_citation']}\n"
            citation_guide += f"   Key content: \"{fg['key_points'][:150]}...\"\n\n"
    
    if articles:
        citation_guide += "SCIENTIFIC LITERATURE:\n"
        formatted_articles = format_pubmed_for_citation(articles, question)
        for fa in formatted_articles[:8]:
            citation_guide += f"{fa['citation_id']} = {fa['study_type']} (PMID: {fa['pmid']})\n"
            citation_guide += f"   Finding: \"{fa['key_finding'][:150]}...\"\n\n"
    
    # Create Q&A-specific system prompt
    qa_system_prompt = f"""
    You are a Medical Information Assistant created by a physician to provide accurate, 
    evidence-based answers to health questions.
    
    {citation_guide}
    
    IMPORTANT GUIDELINES:
    1. Provide comprehensive yet accessible answers
    2. Always cite your sources using [G#] for guidelines and [P#] for PubMed
    3. Include important caveats and limitations
    4. Emphasize this is educational information, not personal medical advice
    5. Suggest when to consult healthcare providers
    
    Structure your response:
    1. **Direct Answer**: Clear, concise answer to the question
    2. **Detailed Explanation**: Evidence-based details with citations
    3. **Important Considerations**: Caveats, risk factors, when to seek care
    4. **Summary**: Key takeaways
    
    Remember to cite sources like: "According to [G1]..." or "A study [P2] found..."
    """
    
    # Format evidence for API call
    pubmed_evidence = ""
    if articles:
        pubmed_evidence = format_pubmed_evidence_with_forced_citations(articles)
    
    # Prepare user message
    user_message = f"Please answer this health question:\n\n{question}"
    
    if pubmed_evidence:
        user_message += f"\n\nRelevant scientific evidence:\n{pubmed_evidence}"
    
    # API call with o3-mini support
    config = MODEL_CONFIGS.get(model, {"temperature": 0.1, "max_tokens": 2000})
    
    messages = [
        {"role": "system", "content": qa_system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    payload = prepare_api_payload(
        model=model,
        messages=messages,
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 2000)
    )
    
    try:
        api_response = make_openai_request(payload, api_key)
        
        if api_response["success"]:
            data = api_response["data"]
            answer = data["choices"][0]["message"]["content"]
            
            # Enhance with links
            if articles:
                answer = enhance_analysis_with_links(answer, articles)
            
            return {
                "success": True,
                "answer": answer,
                "model": model,
                "processing_time": time.time() - start_time,
                "tokens_used": data.get("usage", {}).get("total_tokens"),
                "guidelines_used": len(relevant_guidelines),
                "pubmed_articles": articles,
                "guidelines": relevant_guidelines
            }
        else:
            # Handle o3-mini fallback
            if "o3" in model.lower():
                st.warning(f"⚠️ {api_response['error']}")
                st.info("🔄 Falling back to GPT-4o...")
                
                fallback_payload = prepare_api_payload(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                )
                
                fallback_response = make_openai_request(fallback_payload, api_key)
                if fallback_response["success"]:
                    data = fallback_response["data"]
                    answer = data["choices"][0]["message"]["content"]
                    
                    if articles:
                        answer = enhance_analysis_with_links(answer, articles)
                    
                    return {
                        "success": True,
                        "answer": answer,
                        "model": "gpt-4o (fallback)",
                        "processing_time": time.time() - start_time,
                        "tokens_used": data.get("usage", {}).get("total_tokens"),
                        "guidelines_used": len(relevant_guidelines),
                        "pubmed_articles": articles,
                        "guidelines": relevant_guidelines
                    }
            
            return {
                "success": False,
                "error": api_response["error"],
                "processing_time": time.time() - start_time
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def handle_followup_question(question, conversation_history, api_key):
    """Handle follow-up questions with context"""
    # Build context from previous Q&A
    context = "Previous conversation:\n"
    for qa in conversation_history[-3:]:  # Last 3 exchanges
        context += f"Q: {qa['question']}\nA: {qa['answer'][:200]}...\n\n"
    
    # Detect if this is a follow-up
    if any(word in question.lower() for word in ['that', 'it', 'this', 'more', 'else', 'what about']):
        # Include context in the search and response generation
        enhanced_question = f"{context}\nCurrent question: {question}"
        return answer_health_question(enhanced_question, api_key)
    else:
        return answer_health_question(question, api_key)

def extract_health_claims_from_text(text, api_key):
    """
    Extract individual health claims from user-submitted text using GPT with deterministic results
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    health_claim_extractor_message = """
    You are a medical claim extraction system. Extract EXACTLY the specific, testable health claims from the provided text.

    CRITICAL RULE: PRESERVE ALL SPECIFIC MEDICAL TERMS AND DRUG NAMES
    - If the text mentions "evacetrapib", use "evacetrapib" in the claim
    - If the text mentions "ezetimibe", use "ezetimibe" in the claim  
    - If the text mentions "atorvastatin", use "atorvastatin" in the claim
    - NEVER replace specific drug names with generic terms like "cholesterol drug" or "statin"

    STRICT RULES:
    1. Only extract claims that make specific medical/health assertions
    2. Use the EXACT wording from the text when possible, especially for drug names, medical conditions, and specific measurements
    3. Each claim must be a complete, standalone statement
    4. Order claims by their appearance in the text
    5. Maximum 6 claims total
    6. Minimum 3 claims (combine related statements if needed)
    7. ALWAYS preserve specific drug names, dosages, percentages, and medical terminology
    
    FORMAT REQUIREMENTS:
    - Start each claim with "CLAIM X:" where X is the number
    - Use quotes around the exact claim text
    - One claim per line
    - No additional commentary or explanations
    
    EXAMPLE OUTPUT:
    CLAIM 1: "Evacetrapib lowered LDL levels by 37% but had no effect on heart health"
    CLAIM 2: "Adding ezetimibe to atorvastatin significantly improved lipid profiles"
    CLAIM 3: "Statins help patients after myocardial infarctions regardless of cholesterol levels"
    
    FOCUS ON: Treatment effects, drug outcomes, medical procedures, health interventions, diagnostic claims, causation statements.
    IGNORE: General background information, author opinions, study methodology descriptions.
    
    PRESERVE SPECIFICITY: Always use the exact drug names, medical conditions, measurements, and terminology from the source text.
    """
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": health_claim_extractor_message},
            {"role": "user", "content": text}
        ],
        "temperature": 0,  # Set to 0 for maximum determinism
        "max_tokens": 800  # Increased for longer, more detailed claims
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        extracted_claims = data["choices"][0]["message"]["content"].strip()
        
        # Parse the claims with consistent formatting
        claims = []
        lines = [line.strip() for line in extracted_claims.split('\n') if line.strip()]
        
        for line in lines:
            # Look for "CLAIM X:" format and extract the quoted text
            if line.startswith('CLAIM') and ':' in line:
                # Extract everything after the colon
                claim_text = line.split(':', 1)[1].strip()
                # Remove quotes if present
                if claim_text.startswith('"') and claim_text.endswith('"'):
                    claim_text = claim_text[1:-1]
                elif claim_text.startswith("'") and claim_text.endswith("'"):
                    claim_text = claim_text[1:-1]
                
                if claim_text:  # Only add non-empty claims
                    claims.append(claim_text)
            # Fallback: if no CLAIM format, treat as direct claim
            elif not line.startswith('CLAIM') and len(line) > 10:
                claims.append(line)
        
        # Ensure we have at least some claims (fallback to original parsing if needed)
        if not claims:
            claims = [claim.strip() for claim in extracted_claims.split('\n') if claim.strip() and len(claim.strip()) > 10]
        
        print(f"🔍 DEBUG: Extracted {len(claims)} claims from text")
        for i, claim in enumerate(claims, 1):
            print(f"🔍 DEBUG: Claim {i}: {claim[:100]}...")
        
        return claims
    except Exception as e:
        st.error(f"Error extracting claims: {str(e)}")
        return []

def analyze_claim_with_pubmed(claim, api_key, model="gpt-4o", ncbi_api_key=None, ncbi_email=None):
    """Enhanced analysis with medical accuracy optimizations and triglyceride expertise"""
    import requests
    import json
    import time
    
    start_time = time.time()
    
    # Initialize the enhanced analyzer
    claim_analyzer = MedicalClaimAnalyzer()
    
    # Check if this is a triglyceride-related claim
    triglyceride_indicators = ['triglyceride', 'tg', 'chylomicronemia', 'milky blood', 'lactescent', 'hypertriglyceridemia']
    is_triglyceride_claim = any(indicator in claim.lower() for indicator in triglyceride_indicators)
    triglyceride_context = []
    domain_validation = None
    
    if is_triglyceride_claim:
        # Get expert context
        triglyceride_context = claim_analyzer.triglyceride_expert.generate_expert_context(claim)
        
        # Extract and validate any triglyceride values
        extracted_values = []
        tg_value = claim_analyzer.triglyceride_expert.validate_triglyceride_value(claim)
        if tg_value:
            triglyceride_context.append(f"Triglyceride value {tg_value['value']} mg/dL is classified as: {tg_value['classification']}")
            triglyceride_context.append(f"Clinical action: {tg_value['clinical_action']}")
            extracted_values.append(tg_value)
        
        # Validate intervention claims
        intervention_validations = claim_analyzer.triglyceride_expert.validate_intervention_claim(claim)
        
        # Assess plausibility
        plausibility = claim_analyzer.triglyceride_expert.assess_claim_plausibility(claim, extracted_values)
        
        # Generate expert context
        expert_context = claim_analyzer.triglyceride_expert.generate_expert_context(claim)
        
        domain_validation = {
            'values': extracted_values,
            'interventions': intervention_validations,
            'plausibility': plausibility,
            'expert_context': expert_context
        }
    
    # Step 1: Extract individual health claims
    with st.spinner("Extracting individual health claims..."):
        claims = extract_health_claims_from_text(claim, api_key)
        
        if not claims:
            # Fallback to the original decomposition
            claims = claim_analyzer.decompose_complex_claim(claim)
        
        if len(claims) > 1:
            st.info(f"Identified {len(claims)} distinct claims to analyze")
            with st.expander("View extracted claims"):
                for i, component in enumerate(claims):
                    st.write(f"{i+1}. {component}")
    
    # Step 2: Detect red flags early
    all_red_flags = {}
    for i, individual_claim in enumerate(claims):
        red_flags = claim_analyzer.detect_red_flags(individual_claim)
        if red_flags:
            all_red_flags[i] = red_flags
    
    if all_red_flags:
        st.warning(f"⚠️ Detected potential red flags in {len(all_red_flags)} claims")
        with st.expander("View red flag details"):
            for claim_idx, flags in all_red_flags.items():
                st.write(f"**Claim {claim_idx + 1}:** {claims[claim_idx][:50]}...")
                for flag_type, patterns in flags.items():
                    st.write(f"  - {flag_type.replace('_', ' ').title()}")
    
    # Step 3: Get guidelines for each claim individually
    all_guidelines = []
    claim_guidelines = {}
    
    with st.spinner(f"Finding medical society guidelines for {len(claims)} claims..."):
        for i, individual_claim in enumerate(claims):
            st.info(f"Searching guidelines for claim {i+1}: {individual_claim[:100]}...")
            
            # Get guidelines for this specific claim
            guideline_results = get_relevant_guidelines(individual_claim, max_guidelines=3)
            
            if guideline_results:
                claim_guidelines[i] = guideline_results
                all_guidelines.extend(guideline_results)
            
            # Try expanded queries if no results
            if not guideline_results:
                expanded_queries = expand_medical_query(individual_claim)
                for expanded_query in expanded_queries[1:3]:
                    guideline_results = get_relevant_guidelines(expanded_query, max_guidelines=2)
                    if guideline_results:
                        claim_guidelines[i] = guideline_results
                        all_guidelines.extend(guideline_results)
                        st.success(f"Found guidelines using expanded search: '{expanded_query[:50]}...'")
                        break
    
    # Remove duplicate guidelines
    unique_guidelines = []
    seen_guideline_ids = set()
    for guideline in all_guidelines:
        if guideline['id'] not in seen_guideline_ids:
            unique_guidelines.append(guideline)
            seen_guideline_ids.add(guideline['id'])
    
    formatted_guidelines = []
    if unique_guidelines:
        formatted_guidelines = format_guidelines_for_citation(unique_guidelines, claim)
        st.success(f"✓ Found {len(formatted_guidelines)} relevant medical society guidelines")
    else:
        st.info("No specific medical society guidelines found. The analysis will rely on PubMed evidence.")
    
    # Step 4: Enhanced PubMed search for each individual claim
    pubmed_searcher = PubMedSearcher(email=ncbi_email, api_key=ncbi_api_key)
    all_articles = []
    claim_articles = {}
    
    with st.spinner(f"Searching PubMed for evidence on {len(claims)} health claims..."):
        for i, individual_claim in enumerate(claims):
            st.write(f"Searching for claim {i+1}: {individual_claim[:100]}...")
            
            # Search for this specific claim
            articles = pubmed_searcher.find_relevant_articles(individual_claim, max_results=10)
            if articles:
                claim_articles[i] = articles
                all_articles.extend(articles)
                st.success(f"Found {len(articles)} articles for claim {i+1}")
            else:
                st.warning(f"No articles found for claim {i+1}")
            
            # Add delay between searches
            if i < len(claims) - 1:
                time.sleep(0.5)
    
    # Remove duplicate articles
    unique_articles = []
    seen_pmids = set()
    for article in all_articles:
        if article['pmid'] not in seen_pmids:
            unique_articles.append(article)
            seen_pmids.add(article['pmid'])
    
    # Sort by quality score
    unique_articles.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    # ADD THIS NEW SECTION: Analyze evidence quality
    evidence_analyzer = EvidenceQualityAnalyzer()
    evidence_analysis = None
    
    if unique_articles:
        st.info("🔬 Analyzing evidence quality and effect sizes...")
        evidence_analysis = evidence_analyzer.analyze_multiple_abstracts(unique_articles)
        
        # Show evidence quality summary
        summary = evidence_analysis['summary']
        st.success(f"Evidence Analysis: {summary['total_studies']} studies analyzed")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Confidence", evidence_analysis['overall_confidence'].title())
        with col2:
            st.metric("High Confidence Studies", f"{summary.get('high_confidence_studies', 0)}/{summary['total_studies']}")
        with col3:
            caution_pct = summary.get('cautions_percentage', 0)
            st.metric("Studies with Cautions", f"{caution_pct:.0f}%")

    # Show quality distribution
    if unique_articles:
        high_quality = sum(1 for a in unique_articles if a.get('quality_score', 0) >= 80)
        medium_quality = sum(1 for a in unique_articles if 50 <= a.get('quality_score', 0) < 80)
        st.success(f"Found {len(unique_articles)} unique articles: {high_quality} high-quality, {medium_quality} medium-quality")
    
    # Step 5: Perform plausibility checks for each claim
    individual_plausibility = {}
    for i, individual_claim in enumerate(claims):
        current_guidelines = claim_guidelines.get(i, [])
        current_articles = claim_articles.get(i, [])
        individual_plausibility[i] = claim_analyzer.check_claim_plausibility(
            individual_claim, current_guidelines, current_articles
        )
    
    # Step 6: Create enhanced system message with structured evaluation
    citation_guide = "CITATION REFERENCE GUIDE:\n\n"

    # Add guidelines to citation guide
    if formatted_guidelines:
        citation_guide += "MEDICAL SOCIETY GUIDELINES (USE THESE FIRST):\n"
        for fg in formatted_guidelines:
            citation_guide += f"{fg['citation_id']} = {fg['full_citation']}\n"
            citation_guide += f"   Key content: \"{fg['key_points'][:150]}...\"\n\n"
    else:
        citation_guide += "No medical society guidelines found for this topic.\n\n"
    
    # Format PubMed articles for citation
    formatted_articles = []
    if unique_articles:
        formatted_articles = format_pubmed_for_citation(unique_articles, claim)
        citation_guide += "PUBMED EVIDENCE (CITE THESE BY ID):\n"
        for fa in formatted_articles[:8]:  # Top 8 articles
            citation_guide += f"{fa['citation_id']} = {fa['study_type']} (PMID: {fa['pmid']})\n"
            citation_guide += f"   Title: \"{fa['title'][:100]}...\"\n"
            citation_guide += f"   Finding: \"{fa['key_finding'][:150]}...\"\n\n"
    
    # Add triglyceride expert context if relevant
    triglyceride_section = ""
    clinical_context = ""
    if triglyceride_context:
        triglyceride_section = "\n\nTRIGLYCERIDE EXPERT CONTEXT:\n"
        for context in triglyceride_context:
            triglyceride_section += f"- {context}\n"
    
    if domain_validation:
        clinical_context = f"""
    CLINICAL CONTEXT FOR TRIGLYCERIDES:
    - Normal: <150 mg/dL
    - Borderline high: 150-199 mg/dL
    - High: 200-499 mg/dL
    - Very high: 500-999 mg/dL (pancreatitis risk)
    - Severe: ≥1000 mg/dL (urgent treatment needed)
    
    EVIDENCE-BASED INTERVENTIONS:
    - Dietary: 20-50% reduction possible
    - Weight loss: 20-30% per 10% body weight lost
    - Medications: 30-50% with fibrates/omega-3
    
    EXPERT NOTES:
    {' '.join(domain_validation['expert_context'])}
    """
    
    # Create evidence quality guidance
    evidence_quality_guidance = ""
    if evidence_analysis:
        summary = evidence_analysis['summary']
        confidence = evidence_analysis['overall_confidence']
        
        evidence_quality_guidance = f"""
    CRITICAL EVIDENCE QUALITY ANALYSIS:

    Overall Evidence Confidence: {confidence.upper()}
    - Total studies analyzed: {summary['total_studies']}
    - High confidence studies: {summary.get('high_confidence_studies', 0)}
    - Studies with cautions: {summary.get('studies_with_cautions', 0)} ({summary.get('cautions_percentage', 0):.0f}%)
    - Studies with limitations: {summary.get('studies_with_limitations', 0)} ({summary.get('limitations_percentage', 0):.0f}%)

    DETAILED STUDY ANALYSIS:
    """
        
        for analysis in evidence_analysis['analyses'][:5]:  # Top 5 studies
            evidence_quality_guidance += f"""
    Study (PMID: {analysis['pmid']}):
    - Confidence Level: {analysis['confidence_level'].upper()}
    - Effect Size: {analysis['effect_size']['magnitude'].upper()}
    - Cautionary Language: {', '.join(analysis['cautionary_language']) if analysis['cautionary_language'] else 'None'}
    - Study Limitations: {', '.join(analysis['study_limitations']) if analysis['study_limitations'] else 'None'}
    """

        evidence_quality_guidance += f"""
    SCORING GUIDANCE BASED ON EVIDENCE QUALITY:
    - If overall confidence is "low" or "very_low": Cap credibility scores at 40%
    - If >50% of studies have cautions: Reduce scores by 15-25 points
    - If effect sizes are consistently "small": Reduce scores by 10-15 points
    - If multiple limitations found: Reduce scores by 10-20 points

    YOU MUST factor this evidence quality analysis into your credibility scores!
    """

    base_system_message = f"""
    You are a medical fact-checker with expertise in evidence-based medicine{' and specialized knowledge in triglycerides and lipid disorders' if is_triglyceride_claim else ''}.

    CRITICAL SCORING INSTRUCTION:
    When a claim states "There are no treatments for [condition]" or similar:
    - If treatments DO exist, the claim is FALSE and should receive a LOW credibility score (0-20%)
    - You are scoring the ACCURACY OF THE CLAIM, not the existence of treatments
    - A claim saying "no treatments exist" when treatments DO exist is MISINFORMATION
    
    Example: "There are no treatments for familial chylomicronemia" is FALSE because treatments exist, 
    so it should get a LOW score (not high).

    {evidence_quality_guidance}

    {citation_guide}
    {triglyceride_section}
    {clinical_context}

    MANDATORY CITATION AND EVIDENCE QUALITY RULES:
    1. You MUST cite sources using the IDs above: [G1], [G2] for guidelines; [P1], [P2] for PubMed
    2. Every medical claim you make MUST have at least one citation
    3. When guidelines exist ([G#]), prioritize them over PubMed articles
    4. YOU MUST CONSIDER THE EVIDENCE QUALITY ANALYSIS ABOVE when assigning credibility scores
    5. EXPLICITLY mention when evidence has cautions, limitations, or small effect sizes
    6. Use this format: "According to [G1], chylomicronemia is defined as..."
    7. Or: "A systematic review [P1] found that... however, the authors noted caution is warranted"

    CRITICAL CONTEXT:
    - Individual claims identified: {len(claims)}
    - Red flags detected: {len(all_red_flags)}
    - Guidelines available: {len(formatted_guidelines)}
    - High-quality studies: {sum(1 for a in unique_articles if a.get('quality_score', 0) >= 80)}
    - Evidence confidence level: {evidence_analysis['overall_confidence'].upper() if evidence_analysis else 'NOT ANALYZED'}
    {f"- Triglyceride-related claim detected with specialized validation" if is_triglyceride_claim else ""}

    Your response must follow this structure:

    **Individual Claim Assessments:**

    [Analyze each claim separately with its own credibility score, CONSIDERING THE EVIDENCE QUALITY ANALYSIS]

    Remember: If evidence has cautions, limitations, or small effects, this MUST be reflected in lower credibility scores and explicit mentions in your analysis!
    """
    
    # Step 7: Prepare evidence for API call
    pubmed_evidence = ""
    if unique_articles:
        pubmed_evidence = format_pubmed_evidence_with_forced_citations(unique_articles)
    

    # Step 8: Make API call with enhanced prompt using new helper functions
    def apply_evidence_quality_adjustment(base_score: float, evidence_analysis: Dict[str, Any], articles: List[Dict[str, Any]]) -> float:
        """
        Adjust credibility score based on evidence quality analysis
        """
        if not evidence_analysis or not articles:
            return base_score
        
        adjusted_score = base_score
        summary = evidence_analysis['summary']
        
        # Get analyses for articles relevant to this claim
        relevant_analyses = [a for a in evidence_analysis['analyses'] 
                            if any(article.get('pmid') == a['pmid'] for article in articles)]
        
        if not relevant_analyses:
            return base_score
        
        # Apply overall confidence adjustment
        confidence = evidence_analysis['overall_confidence']
        if confidence == 'very_low':
            adjusted_score = min(adjusted_score, 30)  # Cap at 30%
        elif confidence == 'low':
            adjusted_score = min(adjusted_score, 50)  # Cap at 50%
        elif confidence == 'medium':
            adjusted_score *= 0.9  # Slight reduction
        
        # Apply caution penalty
        cautions_pct = summary.get('cautions_percentage', 0)
        if cautions_pct > 50:
            adjusted_score -= 20
        elif cautions_pct > 25:
            adjusted_score -= 10
        
        # Apply small effect size penalty
        small_effects = sum(1 for a in relevant_analyses if a['effect_size']['magnitude'] == 'small')
        if small_effects > len(relevant_analyses) / 2:  # More than half have small effects
            adjusted_score -= 15
        
        # Apply limitation penalty
        limitations_pct = summary.get('limitations_percentage', 0)
        if limitations_pct > 50:
            adjusted_score -= 15
        elif limitations_pct > 25:
            adjusted_score -= 8
        
        return max(10, min(100, adjusted_score))  # Keep between 10-100

    # Calculate individual credibility scores
    individual_scores = {}
    for i, individual_claim in enumerate(claims):
        claim_red_flags = all_red_flags.get(i, {})
        current_guidelines = claim_guidelines.get(i, [])
        current_articles = claim_articles.get(i, [])
        plausibility = individual_plausibility.get(i, {})
        
        # Check for false "no treatment" claims FIRST
        false_claim_penalty = detect_negative_existence_claims(individual_claim)
        
        # Start with very low score if false claim detected
        if false_claim_penalty > 0:
            base_score = 10  # Start very low
        else:
            base_score = 50  # Normal starting point
        
        # Calculate score with the adjusted base
        if domain_validation and is_triglyceride_claim:
            score = claim_analyzer.calculate_credibility_score_with_domain(individual_claim, {
                'evidence_quality': plausibility.get('evidence_quality', 'none'),
                'guideline_alignment': plausibility.get('guideline_alignment', 'not_applicable'),
                'plausibility_checks': plausibility,
                'red_flags': claim_red_flags,
                'high_quality_evidence': sum(1 for a in current_articles if a.get('quality_score', 0) >= 80)
            }, domain_validation)
        else:
            score = claim_analyzer.calculate_credibility_score(individual_claim, {
                'evidence_quality': plausibility.get('evidence_quality', 'none'),
                'guideline_alignment': plausibility.get('guideline_alignment', 'not_applicable'),
                'plausibility_checks': plausibility,
                'red_flags': claim_red_flags,
                'high_quality_evidence': sum(1 for a in current_articles if a.get('quality_score', 0) >= 80)
            })
        
        # Apply false claim penalty
        if false_claim_penalty > 0:
            score = max(0, min(20, score - false_claim_penalty))  # Cap at 20% for false claims
        
        individual_scores[f"Claim {i+1}"] = score

    user_message = f"""
        Analyze these health claims individually:
        
        {json.dumps([f"Claim {i+1}: {c}" for i, c in enumerate(claims)], indent=2)}
        
        CRITICAL: Some of these claims may state that treatments don't exist when they actually do.
        Such claims are FALSE and should receive LOW credibility scores (under 20%).
        Remember: You're scoring the TRUTH of the claim, not whether treatments exist.
        
        Pre-calculated credibility scores (use as guidance):
        {json.dumps(individual_scores, indent=2)}
    
    {f"Triglyceride Expert Assessment: {json.dumps(domain_validation, indent=2)}" if is_triglyceride_claim and domain_validation else ""}
    
    {pubmed_evidence if pubmed_evidence else "No directly relevant PubMed articles were found. Please analyze based on medical society guidelines and your medical knowledge."}
    
    REMINDER: Analyze EACH claim separately with individual assessments and credibility scores.
    """
    
    # Get model-specific configuration
    config = MODEL_CONFIGS.get(model, {"temperature": 0.1, "max_tokens": 2000})

    # Prepare API payload with o3-mini support
    messages = [
        {"role": "system", "content": base_system_message},
        {"role": "user", "content": user_message}
    ]
    
    payload = prepare_api_payload(
        model=model,
        messages=messages,
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 2000)
    )
    
    try:
        with st.spinner("Performing comprehensive medical accuracy analysis..."):
            # Use the new API request helper
            api_response = make_openai_request(payload, api_key)
            
            if not api_response["success"]:
                # Show user-friendly error and try fallback if o3-mini fails
                if "o3" in model.lower() and api_response.get("status_code") in [400, 403]:
                    st.error(f"⚠️ {api_response['error']}")
                    st.info("🔄 Attempting fallback to GPT-4o...")
                    
                    # Try with gpt-4o as fallback
                    fallback_config = MODEL_CONFIGS.get("gpt-4o", {"temperature": 0.1, "max_tokens": 2000})
                    fallback_payload = prepare_api_payload(
                        model="gpt-4o",
                        messages=messages,
                        temperature=fallback_config["temperature"],
                        max_tokens=fallback_config["max_tokens"]
                    )
                    
                    api_response = make_openai_request(fallback_payload, api_key)
                    if not api_response["success"]:
                        return {
                            "success": False,
                            "error": f"Both o3-mini and fallback failed: {api_response['error']}",
                            "processing_time": time.time() - start_time
                        }
                    else:
                        st.success("✅ Successfully used GPT-4o as fallback")
                else:
                    return {
                        "success": False,
                        "error": api_response["error"],
                        "processing_time": time.time() - start_time
                    }
            
            data = api_response["data"]
            analysis = data["choices"][0]["message"]["content"]
            
            # Ensure citations are present
            analysis = check_and_add_missing_citations(analysis, unique_articles)
            
            # Calculate original overall credibility score as average
            original_overall_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50

            # SAFETY CHECK: Cap scores for claims that deny medical facts
            for i, claim in enumerate(claims):
                claim_lower = claim.lower()
                if any(term in claim_lower for term in [
                    'not cause pancreatitis',
                    'don\'t cause pancreatitis',
                    'doesn\'t cause pancreatitis',
                    'no link', 'no connection'
                ]) and any(term in claim_lower for term in ['chylomicronemia', 'triglyceride']):
                    claim_key = f"Claim {i+1}"
                    if claim_key in individual_scores:
                        individual_scores[claim_key] = min(individual_scores[claim_key], 15)
            # Adjust scores based on AI assessment
            adjusted_individual_scores = claim_analyzer.adjust_score_based_on_assessment(
                analysis, individual_scores
            )
            overall_credibility_score = sum(adjusted_individual_scores.values()) / len(adjusted_individual_scores) if adjusted_individual_scores else 50
            
            # Check for significant score adjustments and warn user
            if abs(original_overall_score - overall_credibility_score) > 20:
                st.warning(f"⚠️ Score adjusted from {int(original_overall_score)}% to {int(overall_credibility_score)}% based on detailed assessment")
            
            # Use adjusted individual scores
            individual_scores = adjusted_individual_scores
            
            # ENHANCED: Update ALL score mentions in analysis text with better debugging
            print(f"🔧 DEBUG: Starting score updates in analysis text...")
            
            for i, (claim_label, final_score) in enumerate(individual_scores.items()):
                claim_number = i + 1
                final_score_int = int(final_score)
                original_analysis = analysis
                
                print(f"🔧 DEBUG: Updating Claim {claim_number} to {final_score_int}%")
                
                # Pattern 1: Look for "Credibility Score: XX%" anywhere in the claim section
                # First, extract just this claim's section (from **Claim X:** to **Claim Y:** or end)
                if claim_number < len(individual_scores):
                    next_claim_pattern = rf'\*\*Claim {claim_number}:(.*?)(?=\*\*Claim {claim_number + 1}:)'
                else:
                    next_claim_pattern = rf'\*\*Claim {claim_number}:(.*?)$'
                
                claim_section_match = re.search(next_claim_pattern, analysis, flags=re.DOTALL)
                if claim_section_match:
                    claim_section = claim_section_match.group(1)
                    print(f"🔧 DEBUG: Found Claim {claim_number} section (length: {len(claim_section)})")
                    
                    # Look for SPECIFIC credibility score patterns only (not general percentages)
                    # Pattern 1: "Credibility Score for Claim X: YY%"
                    # Pattern 2: "Credibility Score: YY%"  
                    # Pattern 3: "**Credibility Score for Claim X:** YY%"
                    credibility_patterns = [
                        rf'(Credibility Score for Claim {claim_number}:\s*)\d{{1,3}}(%)',
                        r'(Credibility Score:\s*)\d{1,3}(%)',
                        rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)\d{{1,3}}(%)'
                    ]
                    
                    updated_section = claim_section
                    patterns_found = 0
                    
                    for pattern in credibility_patterns:
                        matches = re.findall(pattern, updated_section)
                        if matches:
                            patterns_found += len(matches)
                            updated_section = re.sub(pattern, rf'\g<1>{final_score_int}\g<2>', updated_section)
                    
                    print(f"🔧 DEBUG: Found {patterns_found} credibility score patterns")
                    
                    if patterns_found > 0:
                        analysis = analysis.replace(claim_section, updated_section)
                        print(f"🔧 DEBUG: ✅ Updated Claim {claim_number} credibility scores in section")
                    else:
                        print(f"🔧 DEBUG: ❌ No credibility score patterns found in Claim {claim_number} section")
                else:
                    print(f"🔧 DEBUG: ❌ Could not extract Claim {claim_number} section")
                
                # Verify the change happened
                if analysis != original_analysis:
                    print(f"🔧 DEBUG: ✅ Analysis text changed for Claim {claim_number}")
                else:
                    print(f"🔧 DEBUG: ❌ Analysis text unchanged for Claim {claim_number}")
            
            print(f"🔧 DEBUG: Finished score updates")

            # VALIDATION: Check score-assessment alignment for PubMed analysis
            validation_issues = validate_score_assessment_alignment(individual_scores, analysis)

            if validation_issues:
                st.warning("⚠️ Score-assessment inconsistencies detected and corrected:")
                for issue in validation_issues:
                    st.warning(f"- {issue['claim']}: Score adjusted from {issue['score']}% to {issue['suggested_score']}% due to {issue['issue']}")
                    individual_scores[issue['claim']] = issue['suggested_score']
                
                # Recalculate overall score with corrected individual scores
                overall_credibility_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50
                
                # ENHANCED FIX: Update ALL occurrences of scores in the analysis text
                for i, (claim_label, final_score) in enumerate(individual_scores.items()):
                    claim_number = i + 1
                    final_score_int = int(final_score)
                    
                    # Fix 1: Replace score in "Credibility Score: XX" format (without percentage)
                    analysis = re.sub(
                        rf'(Claim {claim_number}.*?Credibility Score:\s*)\d+',
                        rf'\g<1>{final_score_int}',
                        analysis,
                        flags=re.DOTALL | re.IGNORECASE
                    )
                    
                    # Fix 2: Replace score in "Credibility Score: XX%" format (with percentage)
                    analysis = re.sub(
                        rf'(Claim {claim_number}.*?Credibility Score:\s*)\d+%',
                        rf'\g<1>{final_score_int}%',
                        analysis,
                        flags=re.DOTALL | re.IGNORECASE
                    )
                    
                    # Fix 3: Replace formatted versions with asterisks
                    analysis = re.sub(
                        rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)\d+%?',
                        rf'\g<1>{final_score_int}%',
                        analysis,
                        flags=re.IGNORECASE
                    )
                    
                    # Fix 4: Simple "Credibility Score for Claim X:" format
                    analysis = re.sub(
                        rf'(Credibility Score for Claim {claim_number}:\s*)\d+%?',
                        rf'\g<1>{final_score_int}%',
                        analysis,
                        flags=re.IGNORECASE
                    )
                    # Final safety check: Make sure claim-specific credibility scores are correct
                # This catches any format we might have missed, but is claim-specific
                for i, (claim_label, final_score) in enumerate(individual_scores.items()):
                    claim_number = i + 1
                    final_score_int = int(final_score)
                    # Only replace credibility scores within this specific claim's section
                    claim_section_pattern = rf'(\*\*Claim {claim_number}:.*?)(Credibility Score:\s*)\d+(%?)'
                    if claim_number < len(individual_scores):
                        claim_section_pattern += rf'(?=.*?\*\*Claim {claim_number + 1}:)'
                    else:
                        claim_section_pattern += r'(?=.*$)'
                        
                    analysis = re.sub(
                        claim_section_pattern,
                        rf'\g<1>\g<2>{final_score_int}\g<3>',
                        analysis,
                        flags=re.DOTALL
                    )

            # Add quality indicators to the analysis
            quality_summary = f"\n\n**Quality Indicators:**\n"
            quality_summary += f"- Total claims analyzed: {len(claims)}\n"
            quality_summary += f"- Guidelines found: {len(formatted_guidelines)}"
            if len(formatted_guidelines) == 0:
                quality_summary += " ℹ️ (No penalty applied)"
            quality_summary += "\n"
            quality_summary += f"- High-quality evidence: {sum(1 for a in unique_articles if a.get('quality_score', 0) >= 80)}\n"
            quality_summary += f"- Red flags detected: {len(all_red_flags)}\n"
            
            if is_triglyceride_claim and domain_validation:
                quality_summary += "\n**Triglyceride-Specific Assessment:**\n"
                tg_plausibility = domain_validation['plausibility']
                quality_summary += f"- Biological Plausibility: {tg_plausibility['score']}%"
                if tg_plausibility['issues']:
                    quality_summary += f"\n- Issues: {', '.join(tg_plausibility['issues'])}"
            
            enhanced_analysis = analysis + quality_summary
            
            # Enhance analysis with clickable links to PubMed
            enhanced_analysis = enhance_analysis_with_links(enhanced_analysis, unique_articles)
            
            # FINAL SAFETY CHECK: Ensure "no treatment" false claims stay low
            for i, claim in enumerate(claims):
                if detect_negative_existence_claims(claim) > 0:
                    claim_key = f"Claim {i+1}"
                    if claim_key in individual_scores and individual_scores[claim_key] > 20:
                        st.warning(f"Overriding high score for false 'no treatment' claim")
                        individual_scores[claim_key] = 15
                        # Also update in the analysis text
                        analysis = re.sub(
                            rf'(Claim {i+1}.*?Credibility Score:\s*)\d+',
                            rf'\g<1>15',
                            analysis,
                            flags=re.DOTALL | re.IGNORECASE
                        )
            return {
                "success": True,
                "analysis": enhanced_analysis,
                "model": model,
                "processing_time": time.time() - start_time,
                "tokens_used": data.get("usage", {}).get("total_tokens"),
                "credibility_score": overall_credibility_score,
                "individual_scores": individual_scores,
                "pubmed_articles": unique_articles,
                "used_guidelines": formatted_guidelines is not None and len(formatted_guidelines) > 0,
                "red_flags": all_red_flags,
                "plausibility_checks": individual_plausibility,
                "claim_components": claims,
                "is_triglyceride_claim": is_triglyceride_claim,
                "domain_validation": domain_validation
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "processing_time": time.time() - start_time
        }

def analyze_claim_simple(claim, api_key, model="gpt-4o"):
    """Simple claim analysis using direct HTTP request to avoid OpenAI package issues"""
    import requests
    import json
    import time
    
    start_time = time.time()
    
    # Initialize analyzer
    claim_analyzer = MedicalClaimAnalyzer()
    
    # Check if this is a triglyceride-related claim
    triglyceride_indicators = ['triglyceride', 'tg', 'chylomicronemia', 'milky blood', 'lactescent', 'hypertriglyceridemia']
    is_triglyceride_claim = any(indicator in claim.lower() for indicator in triglyceride_indicators)
    triglyceride_context = []
    domain_validation = None
    
    if is_triglyceride_claim:
        # Get expert context
        triglyceride_context = claim_analyzer.triglyceride_expert.generate_expert_context(claim)
        
        # Extract and validate any triglyceride values
        extracted_values = []
        tg_value = claim_analyzer.triglyceride_expert.validate_triglyceride_value(claim)
        if tg_value:
            triglyceride_context.append(f"Triglyceride value {tg_value['value']} mg/dL is classified as: {tg_value['classification']}")
            triglyceride_context.append(f"Clinical action: {tg_value['clinical_action']}")
            extracted_values.append(tg_value)
        
        # Validate intervention claims
        intervention_validations = claim_analyzer.triglyceride_expert.validate_intervention_claim(claim)
        
        # Assess plausibility
        plausibility = claim_analyzer.triglyceride_expert.assess_claim_plausibility(claim, extracted_values)
        
        # Generate expert context
        expert_context = claim_analyzer.triglyceride_expert.generate_expert_context(claim)
        
        domain_validation = {
            'values': extracted_values,
            'interventions': intervention_validations,
            'plausibility': plausibility,
            'expert_context': expert_context
        }
    
    # Extract individual health claims
    with st.spinner("Extracting individual health claims..."):
        claims = extract_health_claims_from_text(claim, api_key)
        
        if not claims:
            # Fallback to the original decomposition
            claims = claim_analyzer.decompose_complex_claim(claim)
    
    # Perform basic checks on each claim
    all_red_flags = {}
    for i, individual_claim in enumerate(claims):
        red_flags = claim_analyzer.detect_red_flags(individual_claim)
        if red_flags:
            all_red_flags[i] = red_flags
    
    # First check for relevant guidelines for each claim
    formatted_guidelines_list = []
    with st.spinner("Finding relevant medical society guidelines..."):
        for i, individual_claim in enumerate(claims):
            _, formatted_guidelines = incorporate_guidelines_into_analysis(individual_claim, "")
            
            if formatted_guidelines:
                formatted_guidelines_list.extend(formatted_guidelines)
        
        # Remove duplicates
        unique_guidelines = []
        seen_ids = set()
        for guideline in formatted_guidelines_list:
            guide_id = f"{guideline['society']}_{guideline['year']}_{guideline['category']}"
            if guide_id not in seen_ids:
                unique_guidelines.append(guideline)
                seen_ids.add(guide_id)
        
        if unique_guidelines:
            st.success(f"Found {len(unique_guidelines)} relevant medical society guidelines")
            
            # Show guidelines in an expander
            with st.expander("View Relevant Guidelines"):
                for i, guideline in enumerate(unique_guidelines):
                    st.markdown(f"### {guideline['society']} ({guideline['year']})")
                    st.markdown(f"**Category:** {guideline['category']}")
                    st.markdown(f"**Quality Score:** {guideline.get('quality_score', 'N/A')}/100")
                    st.markdown(f"**Preview:** {guideline['content_preview']}")
                    st.divider()
        else:
            st.info("No specific medical society guidelines found for these claims.")
            st.write("💡 **Note:** The absence of guidelines doesn't mean the claims are wrong. The analysis will rely on your general medical knowledge.")
    
    # Perform plausibility checks for each claim
    individual_plausibility = {}
    for i, individual_claim in enumerate(claims):
        individual_plausibility[i] = claim_analyzer.check_claim_plausibility(individual_claim, unique_guidelines, None)
    
    # Add triglyceride expert context if relevant
    triglyceride_section = ""
    clinical_context = ""
    if triglyceride_context:
        triglyceride_section = "\n\nTRIGLYCERIDE EXPERT CONTEXT:\n"
        for context in triglyceride_context:
            triglyceride_section += f"- {context}\n"
    
    if domain_validation:
        clinical_context = f"""
    CLINICAL CONTEXT FOR TRIGLYCERIDES:
    - Normal: <150 mg/dL
    - Borderline high: 150-199 mg/dL
    - High: 200-499 mg/dL
    - Very high: 500-999 mg/dL (pancreatitis risk)
    - Severe: ≥1000 mg/dL (urgent treatment needed)
    
    EVIDENCE-BASED INTERVENTIONS:
    - Dietary: 20-50% reduction possible
    - Weight loss: 20-30% per 10% body weight lost
    - Medications: 30-50% with fibrates/omega-3
    
    EXPERT NOTES:
    {' '.join(domain_validation['expert_context'])}
    """
    
    # System message for analysis
    base_system_message = f"""
    You are a Health Information Verification Assistant created by a physician to assess health claims{' with specialized expertise in triglycerides and lipid disorders' if is_triglyceride_claim else ''}.
    Your task is to analyze health claims and provide evidence-based assessments.
    
    CONTEXT:
    - Individual claims identified: {len(claims)}
    - Red flags detected: {len(all_red_flags)}
    - Guidelines available: {len(unique_guidelines)}
    {f"- Triglyceride-related claim detected" if is_triglyceride_claim else ""}
    
    {triglyceride_section}
    {clinical_context}
    
    When analyzing claims:
    1. ANALYZE EACH CLAIM INDIVIDUALLY
    2. ASSESS each claim's accuracy based on current medical knowledge and available evidence
    3. When guidelines exist, prioritize them. When they don't, rely on general medical knowledge
    4. Classify each claim as: "Supported by strong evidence," "Partially supported with caveats," "Insufficient evidence," "Contradicted by evidence," or "Misleading/False"
    5. Explain your assessment using plain language accessible to non-medical audiences
    6. PROVIDE INDIVIDUAL CREDIBILITY SCORES FOR EACH CLAIM

    Use this format for your response:

    **Individual Claim Assessments:**

    **Claim 1:** [Restate the claim]
    **Assessment:** [Choose one: Supported by strong evidence / Partially supported with caveats / Insufficient evidence / Contradicted by evidence / Misleading/False]
    **Evidence-Based Explanation:** [Provide a clear explanation]
    **Relevant Medical Society Guidelines:** [If guidelines are provided, cite them here. If not, note "No specific guidelines found"]
    **Credibility Score for Claim 1:** [0-100%]

    [Repeat for each claim]

    **Overall Analysis Summary:**
    [Synthesize findings across all claims]

    **Bottom Line:** [One sentence summary]
    """
    
    # If we have guidelines, create an enhanced message
    if unique_guidelines:
        guidelines_text = "\n\nRELEVANT MEDICAL SOCIETY GUIDELINES:\n"
        for guideline in unique_guidelines:
            guidelines_text += f"\n{guideline['society']} ({guideline['year']}) - {guideline['category']}:\n"
            guidelines_text += f"{guideline['content_preview']}\n"
        
        base_system_message = guidelines_text + "\n" + base_system_message
    
    # Calculate individual credibility scores
    individual_scores = {}
    for i, individual_claim in enumerate(claims):
        claim_red_flags = all_red_flags.get(i, {})
        plausibility = individual_plausibility.get(i, {})
        
        if domain_validation and is_triglyceride_claim:
            score = claim_analyzer.calculate_credibility_score_with_domain(individual_claim, {
                'evidence_quality': plausibility.get('evidence_quality', 'none'),
                'guideline_alignment': plausibility.get('guideline_alignment', 'not_applicable'),
                'plausibility_checks': plausibility,
                'red_flags': claim_red_flags
            }, domain_validation)
        else:
            score = claim_analyzer.calculate_credibility_score(individual_claim, {
                'evidence_quality': plausibility.get('evidence_quality', 'none'),
                'guideline_alignment': plausibility.get('guideline_alignment', 'not_applicable'),
                'plausibility_checks': plausibility,
                'red_flags': claim_red_flags
            })
        individual_scores[f"Claim {i+1}"] = score
    
    user_message = f"""
    Please analyze these health claims individually:
    
    {json.dumps([f"Claim {i+1}: {c}" for i, c in enumerate(claims)], indent=2)}
    
    Pre-calculated credibility scores (use these in your response):
    {json.dumps(individual_scores, indent=2)}
    """
    
    # Get model-specific configuration and prepare API call
    config = MODEL_CONFIGS.get(model, {"temperature": 0.1, "max_tokens": 1500})
    
    messages = [
        {"role": "system", "content": base_system_message},
        {"role": "user", "content": user_message}
    ]
    
    payload = prepare_api_payload(
        model=model,
        messages=messages,
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 1500)
    )
    
    try:
        with st.spinner("Analyzing health claims..."):
            api_response = make_openai_request(payload, api_key)
            
            if not api_response["success"]:
                # Try fallback for o3-mini if it fails
                if "o3" in model.lower() and api_response.get("status_code") in [400, 403]:
                    st.warning(f"⚠️ {api_response['error']}")
                    st.info("🔄 Falling back to GPT-4o...")
                    
                    fallback_config = MODEL_CONFIGS.get("gpt-4o", {"temperature": 0.1, "max_tokens": 1500})
                    fallback_payload = prepare_api_payload(
                        model="gpt-4o",
                        messages=messages,
                        temperature=fallback_config["temperature"],
                        max_tokens=fallback_config["max_tokens"]
                    )
                    
                    api_response = make_openai_request(fallback_payload, api_key)
                    if not api_response["success"]:
                        return {
                            "success": False,
                            "error": f"Both o3-mini and fallback failed: {api_response['error']}",
                            "processing_time": time.time() - start_time
                        }
                    else:
                        st.success("✅ Successfully used GPT-4o as fallback")
                else:
                    return {
                        "success": False,
                        "error": api_response["error"],
                        "processing_time": time.time() - start_time
                    }
            
            data = api_response["data"]
            analysis = data["choices"][0]["message"]["content"]
        
        # Calculate original overall credibility score as average
        original_overall_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50
        
        # Adjust scores based on AI assessment
        adjusted_individual_scores = claim_analyzer.adjust_score_based_on_assessment(
            analysis, individual_scores
        )
        overall_credibility_score = sum(adjusted_individual_scores.values()) / len(adjusted_individual_scores) if adjusted_individual_scores else 50
        
        # Check for significant score adjustments and warn user
        if abs(original_overall_score - overall_credibility_score) > 20:
            st.warning(f"⚠️ Score adjusted from {int(original_overall_score)}% to {int(overall_credibility_score)}% based on detailed assessment")
        
        # Use adjusted individual scores
        individual_scores = adjusted_individual_scores
        
        # CRITICAL: Update the analysis text to show the same scores as displayed at top
        for i, (claim_label, final_score) in enumerate(individual_scores.items()):
            claim_number = i + 1
            # Find and replace any score mentions in the analysis text
            score_patterns = [
                rf'(Credibility Score for Claim {claim_number}:\s*)(\d+)(%)',
                rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)(\d+)(%)',
            ]
            
            for pattern in score_patterns:
                analysis = re.sub(pattern, rf'\g<1>{int(final_score)}\g<3>', analysis)

        # VALIDATION: Check score-assessment alignment for PubMed analysis
        validation_issues = validate_score_assessment_alignment(individual_scores, analysis)

        if validation_issues:
            st.warning("⚠️ Score-assessment inconsistencies detected and corrected:")
            for issue in validation_issues:
                st.warning(f"- {issue['claim']}: Score adjusted from {issue['score']}% to {issue['suggested_score']}% due to {issue['issue']}")
                individual_scores[issue['claim']] = issue['suggested_score']
            
            # Recalculate overall score with corrected individual scores
            overall_credibility_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50
            
            # Re-synchronize the analysis text with corrected scores
            for i, (claim_label, final_score) in enumerate(individual_scores.items()):
                claim_number = i + 1
                score_patterns = [
                    rf'(Credibility Score for Claim {claim_number}:\s*)(\d+)(%)',
                    rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)(\d+)(%)',
                ]
                
                for pattern in score_patterns:
                    analysis = re.sub(pattern, rf'\g<1>{int(final_score)}\g<3>', analysis)

        return {
            "success": True,
            "analysis": analysis,
            "model": model,
            "processing_time": time.time() - start_time,
            "tokens_used": data.get("usage", {}).get("total_tokens"),
            "credibility_score": overall_credibility_score,
            "individual_scores": individual_scores,
            "used_guidelines": len(unique_guidelines) > 0,
            "red_flags": all_red_flags,
            "claim_components": claims,
            "is_triglyceride_claim": is_triglyceride_claim,
            "domain_validation": domain_validation
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "processing_time": time.time() - start_time
        }

def get_youtube_transcript(video_id, api_key=None):
    """
    YouTube transcript fetcher with Replit-specific handling
    """
    # Detect if running on Replit
    is_replit = os.getenv('REPL_ID') is not None
    
    if is_replit:
        st.warning("⚠️ **Replit Detection**: YouTube may block requests from Replit servers. If transcript fetching fails, consider running this locally or using alternative methods.")
    
    st.info(f"Attempting to fetch transcript for video ID: {video_id}")
    
    # Try importing at the function level
    try:
        import youtube_transcript_api
        st.success("YouTube transcript API module imported successfully")
    except ImportError as e:
        error_msg = "YouTube transcript API not properly installed"
        if is_replit:
            error_msg += " (Replit environment detected - ensure youtube-transcript-api is in requirements.txt)"
        
        st.error(f"Cannot import youtube_transcript_api module: {str(e)}")
        st.error(error_msg)
        st.code("pip install youtube-transcript-api", language="bash")
        return {
            "success": False,
            "error": error_msg,
            "is_replit": is_replit
        }
    
    # Now try to use the API
    try:
        # Use the module directly instead of importing specific classes
        api = youtube_transcript_api.YouTubeTranscriptApi
        
        # First attempt - default method
        try:
            st.info("Attempting default transcript fetch...")
            transcript_list = api.get_transcript(video_id)
            
            if transcript_list:
                # Process the transcript
                transcript_text = ' '.join([entry['text'] for entry in transcript_list])
                transcript_text = ' '.join(transcript_text.split())  # Clean whitespace
                
                st.success(f"Successfully retrieved transcript ({len(transcript_text)} characters)")
                
                return {
                    "success": True,
                    "transcript": transcript_text,
                    "title": f"YouTube Video (ID: {video_id})",
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "method": "default",
                    "length": len(transcript_text),
                    "is_replit": is_replit
                }
                
        except Exception as e:
            error_str = str(e)
            
            # Check for 403 Forbidden errors specifically on Replit
            if "403" in error_str and is_replit:
                st.error("🚫 **403 Forbidden Error on Replit**: YouTube is blocking requests from Replit servers.")
                st.info("💡 **Suggested Solutions**:")
                st.info("• Try running this app locally instead of on Replit")
                st.info("• Use the YouTube Data API with an API key (more reliable)")
                st.info("• Manually copy the transcript from YouTube")
                
                return {
                    "success": False,
                    "error": "YouTube blocked Replit server (403 Forbidden)",
                    "suggest_manual": True,
                    "is_replit": True
                }
            
            st.warning(f"Default method failed: {error_str}")
            
            # Try with language codes
            st.info("Trying alternative language codes...")
            for lang in ['en', 'en-US', 'en-GB', 'a.en']:
                try:
                    transcript_list = api.get_transcript(video_id, languages=[lang])
                    
                    if transcript_list:
                        transcript_text = ' '.join([entry['text'] for entry in transcript_list])
                        transcript_text = ' '.join(transcript_text.split())
                        
                        st.success(f"Successfully retrieved transcript using language: {lang}")
                        
                        return {
                            "success": True,
                            "transcript": transcript_text,
                            "title": f"YouTube Video (ID: {video_id})",
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                            "method": f"language: {lang}",
                            "length": len(transcript_text),
                            "is_replit": is_replit
                        }
                        
                except Exception as lang_error:
                    lang_error_str = str(lang_error)
                    
                    # Check for 403 errors in language attempts too
                    if "403" in lang_error_str and is_replit:
                        st.error("🚫 **403 Forbidden Error on Replit**: YouTube is blocking requests from Replit servers.")
                        return {
                            "success": False,
                            "error": "YouTube blocked Replit server (403 Forbidden)",
                            "suggest_manual": True,
                            "is_replit": True
                        }
                    
                    st.warning(f"Language {lang} failed: {lang_error_str[:50]}...")
                    continue
            
            # If all methods failed
            error_msg = str(e)
            replit_suffix = " (Replit environment detected - YouTube may be blocking server requests)" if is_replit else ""
            
            if "transcript" in error_msg.lower() and "disabled" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Transcripts are disabled for this video" + replit_suffix,
                    "is_replit": is_replit
                }
            elif "unavailable" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Video is unavailable (private, deleted, or region-restricted)" + replit_suffix,
                    "is_replit": is_replit
                }
            else:
                return {
                    "success": False,
                    "error": f"No transcripts found for this video. Original error: {error_msg}" + replit_suffix,
                    "is_replit": is_replit,
                    "suggest_manual": is_replit  # Suggest manual transcript on Replit
                }
                
    except Exception as e:
        error_str = str(e)
        replit_note = ""
        
        if is_replit:
            if "403" in error_str:
                replit_note = " (YouTube blocked Replit server - try running locally)"
            else:
                replit_note = " (Replit environment detected)"
        
        st.error(f"Unexpected error: {type(e).__name__}: {error_str}")
        return {
            "success": False,
            "error": f"Failed to retrieve transcript: {error_str}" + replit_note,
            "is_replit": is_replit,
            "suggest_manual": is_replit and "403" in error_str
        }

def install_or_update_youtube_api():
    """Install or update the youtube-transcript-api package"""
    try:
        import sys
        import subprocess
        
        # First try to uninstall existing version
        st.info("Removing old version of youtube-transcript-api...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "youtube-transcript-api"])
        
        # Install the latest version
        st.info("Installing latest youtube-transcript-api...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "youtube-transcript-api"])
        
        st.success("YouTube Transcript API updated successfully! Please restart the app.")
        return True
    except Exception as e:
        st.error(f"Error installing/updating package: {str(e)}")
        return False

def test_youtube_transcript_functionality():
    """Test function to verify YouTube transcript fetching works"""
    test_videos = {
        "TED Talk (should work)": "https://www.youtube.com/watch?v=8jPQjjsBbIc",
        "YouTube Short": "https://youtube.com/shorts/ZD7GjNpxb_o",
        "Regular Video": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }
    
    results = {}
    for name, url in test_videos.items():
        video_id = extract_video_id(url)
        if video_id:
            result = get_youtube_transcript(video_id)
            results[name] = {
                "video_id": video_id,
                "success": result.get("success", False),
                "method": result.get("method", "N/A"),
                "error": result.get("error", None),
                "transcript_length": result.get("length", 0) if result.get("success") else 0
            }
        else:
            results[name] = {
                "video_id": None,
                "success": False,
                "error": "Could not extract video ID"
            }
    
    return results

def analyze_youtube_transcript_with_pubmed(transcript, api_key, model="gpt-4o", ncbi_api_key=None, ncbi_email=None):
    """Analyze YouTube transcript for health misinformation with PubMed evidence"""
    import requests
    import json
    import time
    import re
    
    start_time = time.time()
    
    # Initialize analyzer
    claim_analyzer = MedicalClaimAnalyzer()
    
    # Step 1: Extract potential health claims from the transcript
    health_claim_extractor_message = """
    Extract the key health claims from this YouTube video transcript. 
    Focus only on specific, testable health claims (not general information).
    For each claim, include the exact quote or a close paraphrase from the transcript.
    Format as a simple list with one claim per line.
    """
    
    # Extract claims
    with st.spinner("Extracting health claims from video transcript..."):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": health_claim_extractor_message},
                {"role": "user", "content": transcript}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API Error during claim extraction: {response.status_code}",
                "processing_time": time.time() - start_time
            }
        
        data = response.json()
        extracted_claims = data["choices"][0]["message"]["content"].strip()
        
        # Parse the claims
        claims = [claim.strip() for claim in extracted_claims.split('\n') if claim.strip()]
        
        if claims:
            st.success(f"Extracted {len(claims)} health claims from the video")
            
            # Display the extracted claims
            with st.expander("View Extracted Health Claims"):
                for i, claim in enumerate(claims):
                    st.write(f"{i+1}. {claim}")
        else:
            st.warning("No specific health claims were identified in this video")
            return {
                "success": True,
                "analysis": "No specific health claims were identified in this video transcript.",
                "model": model,
                "processing_time": time.time() - start_time,
                "claims": []
            }
    
    # Step 2: Analyze each claim for red flags
    all_red_flags = {}
    for i, claim in enumerate(claims):
        red_flags = claim_analyzer.detect_red_flags(claim)
        if red_flags:
            all_red_flags[i] = red_flags
    
    if all_red_flags:
        st.warning(f"⚠️ Red flags detected in {len(all_red_flags)} claims")
    
    # Step 3: Check for relevant guidelines for the claims
    all_guidelines = []
    claim_guidelines = {}
    
    with st.spinner(f"Finding relevant medical society guidelines for {len(claims)} health claims..."):
        for i, claim in enumerate(claims):
            # Get guidelines for this claim
            guideline_results = get_relevant_guidelines(claim, max_guidelines=2)
            
            if guideline_results:
                claim_guidelines[i] = guideline_results
                all_guidelines.extend(guideline_results)
                
                # Show progress
                if (i+1) % 2 == 0 or i == len(claims)-1:
                    st.write(f"Checked guidelines for {i+1}/{len(claims)} claims")
    
    # Remove duplicate guidelines
    unique_guidelines = []
    seen_guideline_ids = set()
    for guideline in all_guidelines:
        if guideline['id'] not in seen_guideline_ids:
            unique_guidelines.append(guideline)
            seen_guideline_ids.add(guideline['id'])
    
    if unique_guidelines:
        st.success(f"Found {len(unique_guidelines)} relevant medical society guidelines")
        
        # Format guidelines for inclusion in the prompt
        guidelines_text = "Relevant Medical Society Guidelines:\n\n"
        for i, guideline in enumerate(unique_guidelines):
            guidelines_text += f"Guideline {i+1}: {guideline['society']} ({guideline['year']})\n"
            guidelines_text += f"Category: {guideline['category']}\n"
            guidelines_text += f"Quality Score: {guideline.get('quality_score', 'N/A')}/100\n"
            
            # Include content (truncated if needed)
            content = guideline.get("content", "")
            if not content and "file_path" in guideline and os.path.exists(guideline["file_path"]):
                with open(guideline["file_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
                
            guidelines_text += f"Content: {content}\n\n"
    else:
        st.info("No specific medical society guidelines found for the claims in this video")
        st.write("💡 **Note:** The absence of guidelines doesn't mean the claims are wrong. The analysis will rely on available scientific evidence.")
        guidelines_text = ""
    
    # Step 4: Search PubMed for evidence related to each claim
    pubmed_searcher = PubMedSearcher(email=ncbi_email, api_key=ncbi_api_key)
    all_articles = []
    claim_articles = {}
    
    # Limit the number of claims to search for (to avoid overwhelming PubMed)
    claims_to_search = claims[:5] if len(claims) > 5 else claims
    
    with st.spinner(f"Searching PubMed for scientific evidence on {len(claims_to_search)} health claims..."):
        st.info("Using enhanced search strategy with compound term preservation")
        
        for i, claim in enumerate(claims_to_search):
            # Use enhanced search for better results
            st.write(f"Searching for claim {i+1}: {claim[:100]}...")
            
            # Use ENHANCED search for YouTube claims - this preserves compound terms!
            claim_articles[i] = pubmed_searcher.find_relevant_articles(claim, max_results=10)
            all_articles.extend(claim_articles[i])
            
            # Show progress 
            st.write(f"Found {len(claim_articles[i])} articles for claim {i+1}")
            
            # Add a longer delay between claim searches to avoid rate limiting
            if i < len(claims_to_search) - 1:
                time.sleep(1.0)  # 1 second between different claims
    
    # Remove duplicate articles
    unique_articles = []
    seen_pmids = set()
    for article in all_articles:
        if article['pmid'] not in seen_pmids:
            unique_articles.append(article)
            seen_pmids.add(article['pmid'])
    
    # Sort by quality score
    unique_articles.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    # Initialize quality counts
    high_quality = 0
    
    if unique_articles:
        # Show quality distribution
        high_quality = sum(1 for a in unique_articles if a.get('quality_score', 0) >= 80)
        medium_quality = sum(1 for a in unique_articles if 50 <= a.get('quality_score', 0) < 80)
        st.success(f"Found {len(unique_articles)} relevant medical articles ({high_quality} high-quality)")
    else:
        st.warning("No specific PubMed evidence found for the claims in this video")
    
    # Format articles for inclusion in the prompt
    pubmed_evidence = format_pubmed_evidence_with_forced_citations(unique_articles)
    
    # Step 5: Analyze the transcript and evidence
    system_message = f"""
    You are a Health Information Verification Assistant created by a physician to assess health claims in video transcripts.
    
    CRITICAL CONTEXT:
    - Total claims identified: {len(claims)}
    - Claims with red flags: {len(all_red_flags)}
    - Guidelines available: {len(unique_guidelines)}
    - High-quality studies: {high_quality}
    
    CRITICAL CITATION REQUIREMENT:
    You MUST cite specific PubMed articles by their PMID numbers when they support or contradict claims.
    For example: "A systematic review (PMID: 12345678) found that..."
    
    IMPORTANT: 
    - PRIORITIZE MEDICAL SOCIETY GUIDELINES OVER PUBMED EVIDENCE
    - Provide individual credibility scores for EACH claim

    For each claim:
    1. ASSESS accuracy using the evidence hierarchy (guidelines > systematic reviews > RCTs > observational)
    2. CHECK for biological plausibility
    3. IDENTIFY red flags (miracle cures, conspiracy language, absolute claims)
    4. CLASSIFY as: "Supported by strong evidence," "Partially supported with caveats," "Insufficient evidence," "Contradicted by evidence," or "Misleading/False"
    5. CITE specific PMIDs when referencing PubMed evidence
    6. PROVIDE A SEPARATE CREDIBILITY SCORE FOR EACH CLAIM
    
    Use this format:

    **Transcript Analysis Summary**

    **Key Health Claims Identified:**
    - Claim 1: [Briefly state the claim]
    - Claim 2: [Briefly state the claim]
    [...]

    **Detailed Assessment:**

    **Claim 1:** "[Direct quote or close paraphrase from transcript]"
    **Assessment:** [Classification]
    **Evidence-Based Explanation:** [Explanation with scientific context and PMID citations]
    **Relevant Guidelines:** [Cite any relevant medical society guidelines]
    **Supporting Literature:** [Cite relevant PubMed articles with PMID]
    **Credibility Score for Claim 1:** [0-100%]

    **Claim 2:** "[Direct quote or close paraphrase from transcript]"
    **Assessment:** [Classification]
    **Evidence-Based Explanation:** [Explanation with scientific context and PMID citations]
    **Relevant Guidelines:** [Cite any relevant medical society guidelines]
    **Supporting Literature:** [Cite relevant PubMed articles with PMID]
    **Credibility Score for Claim 2:** [0-100%]

    [Continue for each claim...]

    **Red Flags:** [List concerning patterns]

    **Bottom Line:** [One or two sentence summary]
    """
    
    with st.spinner("Analyzing video content with medical guidelines and evidence..."):
        # Calculate individual credibility scores for each claim
        individual_scores = {}
        for i, claim in enumerate(claims):
            claim_red_flags = all_red_flags.get(i, {})
            current_claim_guidelines = claim_guidelines.get(i, [])
            claim_evidence = claim_articles.get(i, [])
            
            plausibility = claim_analyzer.check_claim_plausibility(claim, current_claim_guidelines, claim_evidence)
            score = claim_analyzer.calculate_credibility_score(claim, {
                'red_flags': claim_red_flags,
                'guideline_alignment': plausibility['guideline_alignment'],
                'evidence_quality': plausibility['evidence_quality'],
                'plausibility_checks': plausibility
            })
            individual_scores[f"Claim {i+1}"] = score
        
        # Prepare user message
        user_message = f"""
        Please analyze this YouTube video transcript for health claims and misinformation:
        
        {transcript}
        
        Focus on these extracted health claims:
        {extracted_claims}
        
        Red flags detected:
        {json.dumps(all_red_flags, indent=2) if all_red_flags else "None"}
        
        Individual claim credibility scores (pre-calculated):
        {json.dumps(individual_scores, indent=2)}
        
        Assess each claim's accuracy based on medical guidelines and evidence.
        """
        
        # Add guidelines evidence if available
        if guidelines_text:
            user_message += f"\n\nHere are relevant medical society guidelines to help evaluate these claims:\n\n{guidelines_text}"
        
        # Add PubMed evidence if available
        if pubmed_evidence:
            user_message += f"\n\nHere is relevant medical evidence from PubMed to help evaluate these claims:\n\n{pubmed_evidence}"
        else:
            user_message += "\n\nNo specific PubMed evidence was found for these claims. Please evaluate based on the guidelines and your medical knowledge."
        
        user_message += "\n\nREMINDER: You MUST cite specific PMID numbers when discussing evidence AND provide individual credibility scores for each claim."
        
        # Call the API with o3-mini support
        config = MODEL_CONFIGS.get(model, {"temperature": 0.1, "max_tokens": 2500})
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        payload = prepare_api_payload(
            model=model,
            messages=messages,
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 2500)
        )
        
        api_response = make_openai_request(payload, api_key)
        
        if not api_response["success"]:
            # Handle o3-mini fallback
            if "o3" in model.lower():
                st.warning(f"⚠️ {api_response['error']}")
                st.info("🔄 Falling back to GPT-4o...")
                
                fallback_payload = prepare_api_payload(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2500
                )
                
                api_response = make_openai_request(fallback_payload, api_key)
                if not api_response["success"]:
                    return {
                        "success": False,
                        "error": f"Both o3-mini and fallback failed: {api_response['error']}",
                        "processing_time": time.time() - start_time
                    }
                else:
                    st.success("✅ Successfully used GPT-4o as fallback")
            else:
                return {
                    "success": False,
                    "error": api_response["error"],
                    "processing_time": time.time() - start_time
                }
        
        data = api_response["data"]
        analysis = data["choices"][0]["message"]["content"]
        
        # Ensure citations are present for YouTube analysis too
        if unique_articles:
            analysis = check_and_add_missing_citations(analysis, unique_articles)
        
        # Calculate original overall credibility score as average FIRST
        original_overall_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50

        # Apply consistent score adjustment based on AI assessment text
        claim_analyzer = MedicalClaimAnalyzer()
        adjusted_individual_scores = claim_analyzer.adjust_score_based_on_assessment(
            analysis, individual_scores
        )

        # Use the adjusted scores consistently everywhere
        individual_scores = adjusted_individual_scores
        overall_credibility_score = sum(adjusted_individual_scores.values()) / len(adjusted_individual_scores) if adjusted_individual_scores else 50

        # Check for significant score adjustments and warn user
        if abs(original_overall_score - overall_credibility_score) > 20:
            st.warning(f"⚠️ Score adjusted from {int(original_overall_score)}% to {int(overall_credibility_score)}% based on detailed assessment")

        # CRITICAL: Update the analysis text to show the same scores as displayed at top
        for i, (claim_label, final_score) in enumerate(individual_scores.items()):
            claim_number = i + 1
            # Find and replace any score mentions in the analysis text
            score_patterns = [
                rf'(Credibility Score for Claim {claim_number}:\s*)(\d+)(%)',
                rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)(\d+)(%)',
            ]
            
            for pattern in score_patterns:
                analysis = re.sub(pattern, rf'\g<1>{int(final_score)}\g<3>', analysis)

        # STEP 5A: Validate score-assessment alignment
        validation_issues = validate_score_assessment_alignment(individual_scores, analysis)

        if validation_issues:
            st.warning("⚠️ Score-assessment inconsistencies detected and corrected:")
            for issue in validation_issues:
                st.warning(f"- {issue['claim']}: Score adjusted from {issue['score']}% to {issue['suggested_score']}% due to {issue['issue']}")
                individual_scores[issue['claim']] = issue['suggested_score']
            
            # Recalculate overall score with corrected individual scores
            overall_credibility_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50
            
            # Re-synchronize the analysis text with corrected scores
            for i, (claim_label, final_score) in enumerate(individual_scores.items()):
                claim_number = i + 1
                score_patterns = [
                    rf'(Credibility Score for Claim {claim_number}:\s*)(\d+)(%)',
                    rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)(\d+)(%)',
                ]
                
                for pattern in score_patterns:
                    analysis = re.sub(pattern, rf'\g<1>{int(final_score)}\g<3>', analysis)

        # Check for significant score adjustments and warn user
        if abs(original_overall_score - overall_credibility_score) > 20:
            st.warning(f"⚠️ Score adjusted from {int(original_overall_score)}% to {int(overall_credibility_score)}% based on detailed assessment")

        # Use adjusted individual scores
        individual_scores = individual_scores
        
        # Enhance analysis with clickable links to PubMed
        if unique_articles:
            enhanced_analysis = enhance_analysis_with_links(analysis, unique_articles)
        else:
            enhanced_analysis = analysis
    
    return {
        "success": True,
        "analysis": enhanced_analysis,
        "model": model,
        "processing_time": time.time() - start_time,
        "tokens_used": data.get("usage", {}).get("total_tokens"),
        "credibility_score": overall_credibility_score,
        "individual_scores": individual_scores,
        "claims": claims,
        "pubmed_articles": unique_articles,
        "guidelines": unique_guidelines,
        "red_flags": all_red_flags
    }

def analyze_youtube_transcript(transcript, api_key, model="gpt-4o"):
    """Analyze YouTube transcript for health misinformation"""
    import requests
    import json
    import time
    
    start_time = time.time()
    
    # Initialize analyzer
    claim_analyzer = MedicalClaimAnalyzer()
    
    # Extract claims first
    health_claim_extractor_message = """
    Extract the key health claims from this YouTube video transcript. 
    Focus only on specific, testable health claims (not general information).
    For each claim, include the exact quote or a close paraphrase from the transcript.
    Format as a simple list with one claim per line.
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Extract claims
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": health_claim_extractor_message},
            {"role": "user", "content": transcript}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        return {
            "success": False,
            "error": f"API Error during claim extraction: {response.status_code}",
            "processing_time": time.time() - start_time
        }
    
    data = response.json()
    extracted_claims = data["choices"][0]["message"]["content"].strip()
    claims = [claim.strip() for claim in extracted_claims.split('\n') if claim.strip()]
    
    if not claims:
        return {
            "success": True,
            "analysis": "No specific health claims were identified in this video transcript.",
            "model": model,
            "processing_time": time.time() - start_time,
            "claims": []
        }
    
    # Analyze each claim
    individual_scores = {}
    all_red_flags = {}
    
    for i, claim in enumerate(claims):
        red_flags = claim_analyzer.detect_red_flags(claim)
        if red_flags:
            all_red_flags[i] = red_flags
        
        plausibility = claim_analyzer.check_claim_plausibility(claim, None, None)
        score = claim_analyzer.calculate_credibility_score(claim, {
            'red_flags': red_flags,
            'guideline_alignment': plausibility['guideline_alignment'],
            'evidence_quality': plausibility['evidence_quality'],
            'plausibility_checks': plausibility
        })
        individual_scores[f"Claim {i+1}"] = score
    
    # System message for YouTube transcript analysis
    system_message = f"""
    You are a Health Information Verification Assistant created by a physician to assess health claims in video transcripts.
    Your task is to analyze the transcript of a health-related video and identify any potentially misleading or inaccurate health claims.

    CRITICAL: You must provide individual credibility scores for EACH claim identified.

    When analyzing a transcript:
    1. IDENTIFY the key health claims made in the video
    2. ASSESS each claim's accuracy based on current medical evidence
    3. Classify claims as: "Supported by strong evidence," "Partially supported with caveats," "Insufficient evidence," "Contradicted by evidence," or "Misleading/False"
    4. Provide contextual explanations using plain language accessible to non-medical audiences
    5. Note any concerning patterns of misinformation or red flags
    6. PROVIDE A SEPARATE CREDIBILITY SCORE FOR EACH CLAIM

    IMPORTANT: The absence of specific guidelines doesn't automatically make a claim less credible.
    Base your assessment on general medical knowledge and biological plausibility.

    Use this format for your response:

    **Transcript Analysis Summary**

    **Key Health Claims Identified:**
    - Claim 1: [Briefly state the claim]
    - Claim 2: [Briefly state the claim]
    [...]

    **Detailed Assessment:**

    **Claim 1:** "[Direct quote or close paraphrase from transcript]"
    **Assessment:** [Classification]
    **Evidence-Based Explanation:** [Explanation with scientific context]
    **Credibility Score for Claim 1:** [0-100%]

    **Claim 2:** "[Direct quote or close paraphrase from transcript]"
    **Assessment:** [Classification]
    **Evidence-Based Explanation:** [Explanation with scientific context]
    **Credibility Score for Claim 2:** [0-100%]
    [...]

    **Red Flags:** [List any concerning patterns, rhetorical tactics, or claims requiring special attention]

    **Bottom Line:** [One or two sentence summary of the overall accuracy of health information in this video]
    """
    
    # Prepare a user message that includes instructions for handling transcripts
    user_message = f"""
    Please analyze this YouTube video transcript for health claims and misinformation:

    {transcript}

    Focus on identifying specific health claims made in the video and assessing their accuracy based on medical evidence.
    
    Here are the claims I've already identified:
    {json.dumps(claims, indent=2)}
    
    Pre-calculated credibility scores:
    {json.dumps(individual_scores, indent=2)}
    
    Red flags detected:
    {json.dumps(all_red_flags, indent=2) if all_red_flags else "None"}
    """
    
    # Use o3-mini compatible API call
    config = MODEL_CONFIGS.get(model, {"temperature": 0.1, "max_tokens": 2500})
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    payload = prepare_api_payload(
        model=model,
        messages=messages,
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 2500)
    )
    
    try:
        api_response = make_openai_request(payload, api_key)
        
        if not api_response["success"]:
            # Handle o3-mini fallback
            if "o3" in model.lower():
                st.warning(f"⚠️ {api_response['error']}")
                st.info("🔄 Falling back to GPT-4o...")
                
                fallback_payload = prepare_api_payload(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2500
                )
                
                api_response = make_openai_request(fallback_payload, api_key)
                if not api_response["success"]:
                    return {
                        "success": False,
                        "error": f"Both o3-mini and fallback failed: {api_response['error']}",
                        "processing_time": time.time() - start_time
                    }
                else:
                    st.success("✅ Successfully used GPT-4o as fallback")
            else:
                return {
                    "success": False,
                    "error": api_response["error"],
                    "processing_time": time.time() - start_time
                }
        
        data = api_response["data"]
        analysis = data["choices"][0]["message"]["content"]
        
        # Calculate original overall credibility score as average
        original_overall_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50

        # Apply consistent score adjustment based on AI assessment text
        claim_analyzer = MedicalClaimAnalyzer()
        adjusted_individual_scores = claim_analyzer.adjust_score_based_on_assessment(
            analysis, individual_scores
        )

        # Use the adjusted scores consistently everywhere
        individual_scores = adjusted_individual_scores
        overall_credibility_score = sum(adjusted_individual_scores.values()) / len(adjusted_individual_scores) if adjusted_individual_scores else 50

        # Check for significant score adjustments and warn user
        if abs(original_overall_score - overall_credibility_score) > 20:
            st.warning(f"⚠️ Score adjusted from {int(original_overall_score)}% to {int(overall_credibility_score)}% based on detailed assessment")

        # CRITICAL: Update the analysis text to show the same scores as displayed at top
        for i, (claim_label, final_score) in enumerate(individual_scores.items()):
            claim_number = i + 1
            # Find and replace any score mentions in the analysis text
            score_patterns = [
                rf'(Credibility Score for Claim {claim_number}:\s*)(\d+)(%)',
                rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)(\d+)(%)',
            ]
            
            for pattern in score_patterns:
                analysis = re.sub(pattern, rf'\g<1>{int(final_score)}\g<3>', analysis)
        
        # VALIDATION: Check score-assessment alignment for PubMed analysis
        validation_issues = validate_score_assessment_alignment(individual_scores, analysis)

        if validation_issues:
            st.warning("⚠️ Score-assessment inconsistencies detected and corrected:")
            for issue in validation_issues:
                st.warning(f"- {issue['claim']}: Score adjusted from {issue['score']}% to {issue['suggested_score']}% due to {issue['issue']}")
                individual_scores[issue['claim']] = issue['suggested_score']
            
            # Recalculate overall score with corrected individual scores
            overall_credibility_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 50
            
            # Re-synchronize the analysis text with corrected scores
            for i, (claim_label, final_score) in enumerate(individual_scores.items()):
                claim_number = i + 1
                score_patterns = [
                    rf'(Credibility Score for Claim {claim_number}:\s*)(\d+)(%)',
                    rf'(\*\*Credibility Score for Claim {claim_number}:\*\*\s*)(\d+)(%)',
                ]
                
                for pattern in score_patterns:
                    analysis = re.sub(pattern, rf'\g<1>{int(final_score)}\g<3>', analysis)

        return {
            "success": True,
            "analysis": analysis,
            "model": model,
            "processing_time": time.time() - start_time,
            "tokens_used": data.get("usage", {}).get("total_tokens"),
            "credibility_score": overall_credibility_score,
            "individual_scores": individual_scores,
            "claims": claims,
            "red_flags": all_red_flags
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "processing_time": time.time() - start_time
        }

def validate_api_key(api_key):
    """Validate OpenAI API key"""
    import requests
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        return response.status_code == 200
    except:
        return False

def check_replit_storage():
    """
    Check disk usage on Replit and return storage information
    Returns dict with 'is_replit', 'low_storage', 'free_mb', 'used_mb', 'total_mb'
    """
    is_replit = os.getenv('REPL_ID') is not None
    
    if not is_replit:
        return {
            'is_replit': False,
            'low_storage': False,
            'free_mb': None,
            'used_mb': None,
            'total_mb': None
        }
    
    try:
        # Get disk usage for current directory (Replit workspace)
        total, used, free = shutil.disk_usage('.')
        
        # Convert bytes to MB
        total_mb = total // (1024 * 1024)
        used_mb = used // (1024 * 1024)
        free_mb = free // (1024 * 1024)
        
        # Check if free space is less than 100MB
        low_storage = free_mb < 100
        
        return {
            'is_replit': True,
            'low_storage': low_storage,
            'free_mb': free_mb,
            'used_mb': used_mb,
            'total_mb': total_mb
        }
        
    except Exception as e:
        # If there's an error checking disk usage, return safe defaults
        return {
            'is_replit': True,
            'low_storage': False,
            'free_mb': None,
            'used_mb': None,
            'total_mb': None,
            'error': str(e)
        }

def extract_video_id(youtube_url):
    """
    Extract YouTube video ID from various URL formats including shorts
    
    Handles:
    - Standard: https://www.youtube.com/watch?v=VIDEO_ID
    - Short links: https://youtu.be/VIDEO_ID
    - Embeds: https://www.youtube.com/embed/VIDEO_ID
    - Shorts: https://www.youtube.com/shorts/VIDEO_ID
    - Mobile shorts: https://youtube.com/shorts/VIDEO_ID
    - Shorts with query params: https://youtube.com/shorts/VIDEO_ID?feature=share
    """
    import re
    from urllib.parse import urlparse, parse_qs
    
    if not youtube_url:
        return None
        
    # First, check for shorts format specifically
    shorts_patterns = [
        r'(?:youtube\.com\/shorts\/)([\w-]+)',
        r'(?:youtu\.be\/shorts\/)([\w-]+)'
    ]
    
    for pattern in shorts_patterns:
        match = re.search(pattern, youtube_url)
        if match:
            # Found a shorts link
            return match.group(1)
    
    # Common YouTube URL patterns for regular videos
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/|youtube\.com\/v\/)([\w-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    # If patterns don't match, try parsing the URL more carefully
    parsed_url = urlparse(youtube_url)
    
    # Check for regular watch URLs
    if 'youtube.com' in parsed_url.netloc and parsed_url.path == '/watch':
        video_id = parse_qs(parsed_url.query).get('v')
        if video_id:
            return video_id[0]
    
    # Check for shorts directly in the path
    if ('youtube.com' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc) and '/shorts/' in parsed_url.path:
        # Extract the ID from /shorts/VIDEO_ID
        path_parts = parsed_url.path.split('/')
        # Find the index of 'shorts' and get the next part
        try:
            shorts_index = path_parts.index('shorts')
            if shorts_index < len(path_parts) - 1:
                # Return the part after 'shorts/', removing any query parameters
                return path_parts[shorts_index + 1].split('?')[0]
        except ValueError:
            pass  # 'shorts' not found in path
    
    # If we reach here, we couldn't extract the video ID
    return None

def add_guidelines_tab():
    """Add a Guidelines Management tab to the Streamlit app"""
    # Check for PDF support and offer installation if needed
    if not pdf_support:
        st.warning("PDF support is not enabled. Install required packages to process PDF guidelines.")
        if st.button("Install PDF Support", key="install_pdf"):
            install_pdf_support()
    
    # Check for OCR support and offer installation if needed
    if pdf_support and not ocr_support:
        if st.button("Install OCR Support (for scanned PDFs)", key="install_ocr"):
            install_ocr_support()
    
    # Check for Pinecone support
    if not pinecone_support:
        st.warning("Pinecone support is not enabled. Install for semantic search capabilities.")
        if st.button("Install Pinecone Support", key="install_pinecone"):
            install_pinecone_support()
    
    # Upload new guidelines section
    st.subheader("Upload New Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        society_name = st.text_input("Society Name", 
                                   placeholder="e.g., American Heart Association")
        
        category = st.selectbox(
            "Category",
            ["Cardiology", "Endocrinology", "Neurology", "Oncology", 
             "Nutrition", "Pediatrics", "Psychiatry", "General Medicine", "Other"]
        )
    
    with col2:
        year = st.number_input("Publication Year", 
                             min_value=1900, 
                             max_value=datetime.now().year, 
                             value=datetime.now().year)
        
        description = st.text_area("Brief Description", 
                                placeholder="Brief description of the guideline contents...",
                                help="Optional short description to help identify these guidelines")
    
    uploaded_file = st.file_uploader("Upload Guideline File", 
                                  type=["txt", "pdf"], 
                                  help="Upload a text or PDF file containing the guideline details")
    
    if uploaded_file is not None:
        if st.button("Save Guideline", type="primary"):
            with st.spinner("Processing guideline file..."):
                result = upload_guideline(
                    uploaded_file, 
                    society_name, 
                    category, 
                    year,
                    description
                )
                
                if result.get("success"):
                    # Show success message with details
                    st.success("Guideline uploaded successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if "page_count" in result:
                            st.info(f"Detected {result['page_count']} pages")
                        st.info(f"Extracted approximately {result.get('word_count', 'unknown')} words")
                    
                    with col2:
                        st.info(f"Preview: {result.get('preview', '')}")
                    
                    with col3:
                        if result.get("pinecone_uploaded"):
                            st.success("✅ Uploaded to Pinecone for semantic search")
                        else:
                            st.warning("⚠️ Not uploaded to Pinecone")
                    
                    st.experimental_rerun()
                else:
                    st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
    
    # Display existing guidelines
    st.subheader("Existing Guidelines")
    
    guidelines = get_all_guidelines()
    
    if not guidelines:
        st.info("No guidelines have been uploaded yet.")
    else:
        # Create a dataframe for better display
        import pandas as pd
        
        # Extract key information for the table
        guidelines_data = [{
            "Society": g["society"],
            "Category": g["category"],
            "Year": g["year"],
            "Source": g.get("source_type", "TXT"),
            "Upload Date": g["upload_date"],
            "Description": g.get("description", "")[:30] + "..." if g.get("description") and len(g.get("description", "")) > 30 else g.get("description", "")
        } for g in guidelines]
        
        df = pd.DataFrame(guidelines_data)
        st.dataframe(df, use_container_width=True)
        
        # Allow viewing and deletion
        if guidelines:
            col1, col2 = st.columns(2)
            
            with col1:
                # Build selection options with better formatting
                guideline_options = [
                    f"{g['society']} ({g['year']}) - {g['category']}" 
                    for g in guidelines
                ]
                
                selected_guideline_str = st.selectbox(
                    "Select guideline to view or delete:",
                    options=guideline_options,
                    key="guideline_selector"
                )
                
                if selected_guideline_str:
                    # Find the selected guideline
                    selected_index = guideline_options.index(selected_guideline_str)
                    selected_guideline = guidelines[selected_index]
            
            with col2:
                action = st.radio(
                    "Select action:",
                    ["View Content", "View PDF" if pdf_support else "View Content", "Delete Guideline"],
                    horizontal=True,
                    key="guideline_action"
                )
            
            if action == "View Content" and selected_guideline_str:
                content = get_guideline_content(selected_guideline["id"])
                st.subheader(f"Guideline Content: {selected_guideline['society']} ({selected_guideline['year']})")
                st.text_area("", content, height=300)
            
            elif action == "View PDF" and selected_guideline_str and selected_guideline.get("pdf_path"):
                st.subheader(f"PDF Preview: {selected_guideline['society']} ({selected_guideline['year']})")
                
                if os.path.exists(selected_guideline["pdf_path"]):
                    # Create PDF preview
                    pdf_html = create_pdf_preview_html(selected_guideline["pdf_path"])
                    st.components.v1.html(pdf_html, height=500)
                else:
                    st.error("PDF file not found.")
            
            elif action == "Delete Guideline" and selected_guideline_str:
                if st.button("Confirm Delete", type="primary", key="confirm_delete"):
                    if delete_guideline(selected_guideline["id"]):
                        st.success("Guideline deleted successfully")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to delete guideline")
    
    # Sample guideline format section
    with st.expander("How to Format Your Guidelines"):
        st.markdown("""
        ## PDF Guidelines
        
        You can now upload guidelines in PDF format! The system will automatically extract text from:
        
        - Digital PDFs (with selectable text)
        - Scanned PDFs (if OCR support is installed)
        
        For best results:
        - Use PDFs with selectable text rather than scanned images
        - Ensure PDF files don't have strict security settings
        - For scanned PDFs, make sure the text is clear and high-resolution
        
        ## Text File Format
        
        If you're creating a text file, use this recommended format:
        
        ```
        Title: [Title of the Guideline]
        Society: [Medical Society Name]
        Year: [Publication Year]
        
        [Main guideline content with key recommendations]
        
        Key Recommendations:
        1. [First recommendation]
        2. [Second recommendation]
        ...
        
        Evidence Level: [A/B/C or similar rating if available]
        ```
        """)

# Sidebar
with st.sidebar:
    st.title("⚕️ Settings")
    
    # API Key inputs
    st.subheader("API Keys")
    
    # Check if running on Replit and get environment variables
    is_replit = os.getenv('REPL_ID') is not None
    
    # Get default values from environment if on Replit
    default_openai_key = ""
    default_pinecone_key = ""
    default_ncbi_key = ""
    
    if is_replit:
        default_openai_key = os.getenv('OPENAI_API_KEY', '')
        default_pinecone_key = os.getenv('PINECONE_API_KEY', '')
        default_ncbi_key = os.getenv('NCBI_API_KEY', '')
        
        if any([default_openai_key, default_pinecone_key, default_ncbi_key]):
            st.info("🔧 **Replit Detected**: Found API keys in environment variables!")
            
        # Check Replit storage space
        storage_info = check_replit_storage()
        if storage_info.get('low_storage', False):
            free_mb = storage_info.get('free_mb', 0)
            total_mb = storage_info.get('total_mb', 0)
            used_mb = storage_info.get('used_mb', 0)
            
            st.warning(f"⚠️ **Low Storage Warning**: Only {free_mb}MB free space remaining!")
            st.error(f"💾 **Storage Status**: {used_mb}MB used / {total_mb}MB total")
            st.info("""
            **To free up space:**
            • Delete old analysis history files
            • Remove unused uploaded PDFs
            • Clear browser cache and restart Repl
            • Consider upgrading to Replit Pro for more storage
            """)
        elif storage_info.get('free_mb') is not None:
            # Show storage info even when not low (but only if we can read it)
            free_mb = storage_info.get('free_mb', 0)
            total_mb = storage_info.get('total_mb', 0)
            if free_mb > 0:  # Only show if we have valid data
                st.success(f"💾 **Storage**: {free_mb}MB free / {total_mb}MB total")
    
    # OpenAI API key
    openai_api_key = st.text_input("OpenAI API Key", type="password", 
                            value=default_openai_key,
                            help="Enter your OpenAI API key. It will be stored only for this session." + 
                                 (" On Replit, set OPENAI_API_KEY environment variable." if is_replit else ""))
    if openai_api_key:
        st.session_state['openai_api_key'] = openai_api_key
    
    # Pinecone API key
    pinecone_api_key = st.text_input("Pinecone API Key (optional)", type="password",
                           value=default_pinecone_key,
                           help="Enter your Pinecone API key for semantic search capabilities." +
                                (" On Replit, set PINECONE_API_KEY environment variable." if is_replit else ""))
    if pinecone_api_key:
        st.session_state['pinecone_api_key'] = pinecone_api_key
    
    # YouTube API key
    youtube_api_key = st.text_input("YouTube API Key (optional)", type="password",
                           help="Enter your YouTube Data API v3 key for enhanced video analysis. Get one at https://console.cloud.google.com/" +
                                (" On Replit, set YOUTUBE_API_KEY environment variable." if is_replit else ""))
    if youtube_api_key:
        st.session_state['youtube_api_key'] = youtube_api_key
    
    # NCBI API key (new)
    ncbi_api_key = st.text_input("NCBI API Key (optional)", type="password",
                           value=default_ncbi_key,
                           help="Enter your NCBI API key for higher PubMed rate limits. Get one at https://www.ncbi.nlm.nih.gov/account/settings/" +
                                (" On Replit, set NCBI_API_KEY environment variable." if is_replit else ""))
    
    # Email for NCBI (recommended)
    ncbi_email = st.text_input("Email for NCBI (recommended)", 
                          placeholder="your.email@example.com",
                          help="Email address to include with NCBI requests (recommended by NCBI)")
    
    if openai_api_key:
        # Test OpenAI API key
        with st.spinner("Validating OpenAI API key..."):
            if validate_api_key(openai_api_key):
                st.success("✅ OpenAI API key is valid")
            else:
                st.error("❌ Invalid OpenAI API key")
    
    # Advanced options
    with st.expander("Advanced Options"):
        # Detect available models if API key is provided
        if openai_api_key:
            with st.spinner("Detecting available models..."):
                available_models = get_available_models(openai_api_key)
        else:
            available_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        
        # Create model display names with descriptions
        model_display_names = []
        for model in available_models:
            config = MODEL_CONFIGS.get(model, {"description": "Standard model"})
            display_name = f"{model} - {config['description']}"
            model_display_names.append(display_name)
        
        # Model selection
        selected_display = st.selectbox(
            "Model",
            model_display_names,
            index=0,
            help="Models available to your API key. O3-mini offers enhanced reasoning capabilities but may take longer to respond."
        )
        
        # Extract actual model name from display name
        model_option = selected_display.split(" - ")[0]
        
        # Show model details with special handling for o3-mini
        config = MODEL_CONFIGS.get(model_option, {})
        if config:
            if model_option == "o3-mini":
                st.info(f"🧠 Using {model_option}: Enhanced reasoning mode, Max tokens={config['max_tokens']}")
                st.warning("⏱️ Note: o3-mini responses take longer due to advanced reasoning")
                if config.get("special_handling"):
                    st.info("🔧 Special API handling enabled for o3-series model")
            else:
                st.info(f"Using {model_option}: Temperature={config.get('temperature', 'N/A')}, Max tokens={config.get('max_tokens', 'N/A')}")
        else:
            st.warning(f"⚠️ Unknown model configuration for {model_option}")
            
        # Store the selected model for use throughout the app
        st.session_state['selected_model'] = model_option
        
        use_pubmed = st.checkbox("Use PubMed for evidence", value=True, 
                               help="Search PubMed for scientific evidence related to the health claim")
        
        # Updated YouTube API installation section
        st.subheader("YouTube Transcript API")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Install/Update YouTube API"):
                install_or_update_youtube_api()
        with col2:
            if st.button("Test YouTube API"):
                with st.spinner("Testing YouTube transcript functionality..."):
                    test_results = test_youtube_transcript_functionality()
                    for video_name, result in test_results.items():
                        if result["success"]:
                            st.success(f"✅ {video_name}: Working (Method: {result['method']})")
                        else:
                            st.error(f"❌ {video_name}: {result['error']}")
    
    # PDF Support section
    with st.expander("PDF Support"):
        st.write("PDF support status:")
        
        if pdf_support:
            st.success("✅ PDF support is enabled")
        else:
            st.error("❌ PDF support is not enabled")
            if st.button("Install PDF Support", key="sidebar_install_pdf"):
                install_pdf_support()
        
        if ocr_support:
            st.success("✅ OCR support is enabled")
        else:
            st.warning("⚠️ OCR support is not enabled (needed for scanned PDFs)")
            if st.button("Install OCR Support", key="sidebar_install_ocr"):
                install_ocr_support()
    
    # Pinecone Support section
    with st.expander("Semantic Search"):
        st.write("Pinecone support status:")
        
        if pinecone_support:
            st.success("✅ Pinecone support is enabled")
            if pinecone_api_key:
                pinecone_index = init_pinecone()
                if pinecone_index:
                    st.success("✅ Connected to Pinecone")
                else:
                    st.error("❌ Could not connect to Pinecone")
            else:
                st.warning("⚠️ Enter Pinecone API key to enable semantic search")
        else:
            st.error("❌ Pinecone support is not enabled")
            if st.button("Install Pinecone Support", key="sidebar_install_pinecone"):
                install_pinecone_support()
        # ADD THIS DEBUG BUTTON
        if st.button("Debug Pinecone Connection"):
            debug_pinecone_connection()
            
    # About section
    with st.expander("About"):
        st.markdown("""
        **Evidence-based Health Misinformation Checker & Q&A Assistant**
        
        This tool analyzes health claims and answers health questions using medical society guidelines as the primary authority when available, supplemented by PubMed evidence when needed.
        
        **Key Features:**
        - **NEW: Health Q&A Mode** - Ask any health question
        - Semantic search with Pinecone vector database
        - Prioritizes medical society guidelines when available
        - Enhanced PubMed search with compound term preservation
        - Detects red flags and misinformation patterns
        - Provides transparent credibility scoring
        - Fair scoring: No penalty for lacking guidelines
        - Individual credibility scores for each claim
        - Specialized triglyceride expertise
        
        **Scoring Philosophy:**
        - Claims WITH supporting guidelines get bonus points
        - Claims WITHOUT guidelines rely on evidence quality
        - Prevents bias against newer health topics
        - Each claim receives its own credibility score
        - Triglyceride claims validated against clinical reference ranges
        
        **Created by:** A physician  
        **Version:** 4.0 (Q&A Enhanced Edition)
        
        **Disclaimer:** This tool is for educational purposes only and does not provide medical advice.
        """)

# Main area
st.title("🩺 Evidence-based Health Information Assistant")
st.markdown("**Enhanced with Semantic Search, Q&A Capabilities, and Triglyceride Expertise**")
st.markdown("*Fair scoring: Claims without guidelines aren't penalized*")

# Initialize conversation history in session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analyze Content", "Ask a Question", "Guidelines", "History", "Examples"])

with tab1:
    # Create subtabs for text and YouTube
    text_tab, youtube_tab = st.tabs(["Analyze Text", "Analyze YouTube Video"])
    
    with text_tab:
        st.write("Paste social media text about skin care, supplements, diets, or exercise. The AI will verify accuracy based on medical society guidelines and scientific knowledge.")
        
        # Example selector
        example_options = {
            "None": "",
            "Omega-3 Fatty Acids": "Omega-3 fatty acids can lower triglycerides, help regulate blood pressure, and decrease the risk of heart disease.",
            "Celery Juice": "Drinking celery juice every morning on an empty stomach can detoxify your liver, reduce inflammation, and cure autoimmune diseases by flushing toxins from your body.",
            "Vitamin D": "Taking 10,000 IU of Vitamin D daily will boost your immune system and prevent all respiratory infections. Big Pharma doesn't want you to know this simple trick!",
            "Skincare": "This natural face oil penetrates all seven layers of skin to repair DNA damage and reverse aging at the cellular level. It's chemical-free and works better than any prescription.",
            "Weight Loss": "Drinking apple cider vinegar before meals blocks 90% of carb absorption and melts belly fat overnight without any exercise or diet changes.",
            "Cancer Cure": "Alkaline water with lemon cures cancer in 30 days by changing your body's pH. Doctors won't tell you because chemotherapy is more profitable.",
            "High Triglycerides": "My triglycerides are 850 mg/dL. Taking fish oil supplements can lower them by 70% in just 3 days without any dietary changes.",
            "Triglyceride Denial": "High triglycerides do not cause pancreatitis. The medical establishment is lying to sell more medications.",
            "Chylomicronemia False Claim": "Chylomicronemia does not cause pancreatitis. This is a myth perpetuated by pharmaceutical companies."
        }
        
        example = st.selectbox("Try an example:", options=list(example_options.keys()), key="text_example")
        
        # Text input area with example text if selected
        default_text = example_options[example] if example != "None" else ""
        user_text = st.text_area("Enter social media content here:", value=default_text, height=200, 
                              placeholder="Paste or type the health claim here...")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            check_button = st.button("⚕️ Analyze Claim", type="primary", use_container_width=True, key="analyze_text")
        
        with col2:
            api_status = ""
            if not openai_api_key:
                api_status = "⚠️ API key required"
            
        st.text(api_status)
        
        # Process analysis when button is clicked
        if check_button:
            if not openai_api_key:
                st.error("⚠️ OpenAI API Key not configured!")
                st.warning(
                    "Please enter your OpenAI API key in the sidebar to use this application. "
                    "Your key is required to analyze health claims using GPT models."
                )
            elif not user_text.strip():
                st.warning("Please enter some text to analyze!")
            else:
                try:
                    # Choose analysis method based on whether to use PubMed
                    if use_pubmed:
                        # Use PubMed-enhanced analysis with NCBI API key if available
                        selected_model = st.session_state.get('selected_model', model_option)
                        result = analyze_claim_with_pubmed(
                            user_text, 
                            openai_api_key, 
                            model=selected_model,
                            ncbi_api_key=ncbi_api_key,
                            ncbi_email=ncbi_email
                        )
                    else:
                        # Use simple analysis without PubMed
                        selected_model = st.session_state.get('selected_model', model_option)
                        result = analyze_claim_simple(user_text, openai_api_key, model=selected_model)
                    
                    if result["success"]:
                        # Display results
                        st.subheader("Analysis Results:")
                        
                        # Show individual credibility scores if available
                        if result.get("individual_scores"):
                            st.markdown("### Individual Claim Credibility Scores:")
                            
                            # Initialize claim analyzer for explanations
                            claim_analyzer = MedicalClaimAnalyzer()
                            
                            # Create a visual representation of individual scores
                            for claim_label, score in result["individual_scores"].items():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    if score >= 70:
                                        st.success(f"**{claim_label}: {score}%** ✅")
                                    elif score >= 40:
                                        st.warning(f"**{claim_label}: {score}%** ⚠️")
                                    else:
                                        st.error(f"**{claim_label}: {score}%** ❌")
                                with col2:
                                    st.progress(score / 100)
                                
                                # Add detailed explanation for scores below 85%
                                if score < 85:
                                    with st.expander(f"🔍 Why did {claim_label} get {int(score)}%?", expanded=False):
                                        try:
                                            # Extract claim text for this specific claim
                                            claim_number = int(claim_label.split()[-1]) - 1
                                            claims = result.get("claims", [user_text])  # Fallback to full text
                                            claim_text = claims[claim_number] if claim_number < len(claims) else user_text
                                            
                                            # Get analysis results for this claim
                                            analysis_results = {
                                                'red_flags': result.get("red_flags", {}),
                                                'evidence_quality': result.get("evidence_quality", "none"),
                                                'guideline_alignment': result.get("guideline_alignment", "not_applicable"),
                                                'plausibility_checks': result.get("plausibility_checks", {}),
                                                'high_quality_evidence': result.get("high_quality_evidence", 0)
                                            }
                                            
                                            # Generate detailed explanation
                                            explanation_lines = claim_analyzer.generate_score_explanation(
                                                claim_text, score, analysis_results, 
                                                domain_validation=result.get("domain_validation")
                                            )
                                            
                                            # Display explanation
                                            for line in explanation_lines:
                                                if line.strip():
                                                    st.markdown(line)
                                                else:
                                                    st.markdown("")
                                        except Exception as e:
                                            st.markdown("📋 **Score factors:**")
                                            st.markdown("• This score was calculated based on evidence quality, guideline alignment, red flags, and biological plausibility")
                                            st.markdown("• Scores below 85% indicate areas for improvement such as adding peer-reviewed sources, explaining mechanisms, or addressing red flags")
                                            st.markdown("• See the detailed analysis below for specific feedback")
                        
                        # Show overall credibility score
                        st.markdown("### Overall Credibility Score:")
                        if result.get("credibility_score") is not None:
                            score = int(result["credibility_score"])
                            if score >= 70:
                                st.success(f"**{score}%** ✅")
                            elif score >= 40:
                                st.warning(f"**{score}%** ⚠️")
                            else:
                                st.error(f"**{score}%** ❌")
                            
                            # Add note about guideline availability
                            if not result.get("used_guidelines"):
                                st.info("ℹ️ No specific medical society guidelines were found for this claim. The score is based on available scientific evidence.")
                            
                            # Add note if triglyceride claim
                            if result.get("is_triglyceride_claim"):
                                st.info("🔬 Triglyceride-specific expertise was applied to this analysis.")
                        
                        # Display the analysis
                        st.markdown(result["analysis"])
                        
                        # Show metadata
                        with st.expander("Analysis Metadata"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                            with col2:
                                st.metric("Model Used", result["model"])
                            with col3:
                                if result.get("tokens_used"):
                                    st.metric("Tokens Used", result["tokens_used"])
                            with col4:
                                guidelines_status = "Not Found" if not result.get("used_guidelines") else "Yes"
                                st.metric("Guidelines Used", guidelines_status)
                        
                        # Show decomposed claims if available
                        if result.get("claim_components") and len(result["claim_components"]) > 1:
                            with st.expander("View Claim Components"):
                                st.write("The claim was broken down into these testable components:")
                                for i, component in enumerate(result["claim_components"]):
                                    st.write(f"{i+1}. {component}")
                        
                        # Show red flags if detected
                        if result.get("red_flags"):
                            with st.expander("View Detected Red Flags"):
                                st.warning("The following concerning patterns were detected:")
                                for claim_idx, flags in result["red_flags"].items():
                                    if isinstance(claim_idx, int):
                                        st.write(f"**Claim {claim_idx + 1}:**")
                                    for flag_type, patterns in flags.items():
                                        st.write(f"  - {flag_type.replace('_', ' ').title()}")
                        
                        # Show domain validation if triglyceride claim
                        if result.get("domain_validation") and result.get("is_triglyceride_claim"):
                            with st.expander("View Triglyceride Expert Assessment"):
                                domain_val = result["domain_validation"]
                                
                                # Show extracted values
                                if domain_val.get("values"):
                                    st.subheader("Detected Values")
                                    for value in domain_val["values"]:
                                        st.info(f"Triglyceride: {value['value']} mg/dL - {value['classification']}")
                                        st.info(f"Clinical Action: {value['clinical_action']}")
                                
                                # Show intervention validation
                                if domain_val.get("interventions"):
                                    st.subheader("Intervention Analysis")
                                    for intervention in domain_val["interventions"]:
                                        if "expected_reduction" in intervention:
                                            st.success(f"{intervention['intervention']}: {intervention['expected_reduction']} reduction expected")
                                        elif "issue" in intervention:
                                            st.error(f"Issue: {intervention['explanation']}")
                                
                                # Show plausibility
                                if domain_val.get("plausibility"):
                                    plaus = domain_val["plausibility"]
                                    st.subheader("Biological Plausibility")
                                    st.metric("Score", f"{plaus['score']}%")
                                    if plaus.get("issues"):
                                        st.error("Issues: " + ", ".join(plaus["issues"]))
                        
                        # Show PubMed articles if available
                        if use_pubmed and result.get("pubmed_articles"):
                            with st.expander("View PubMed Sources"):
                                # Group by quality
                                high_q = [a for a in result["pubmed_articles"] if a.get('quality_score', 0) >= 80]
                                med_q = [a for a in result["pubmed_articles"] if 50 <= a.get('quality_score', 0) < 80]
                                
                                if high_q:
                                    st.markdown("### High-Quality Evidence")
                                    for article in high_q:
                                        st.markdown(f"**{article['title']}**")
                                        st.markdown(f"Quality Score: {article.get('quality_score')}/100")
                                        st.markdown(f"Type: {', '.join(article.get('publication_types', ['Unknown']))}")
                                        st.markdown(f"Journal: {article['journal']}, {article['year']}")
                                        if article.get('pubmed_url'):
                                            st.markdown(f"[View on PubMed]({article['pubmed_url']})")
                                        st.divider()
                                
                                if med_q:
                                    st.markdown("### Medium-Quality Evidence")
                                    for article in med_q:
                                        st.markdown(f"**{article['title']}**")
                                        st.markdown(f"Quality Score: {article.get('quality_score')}/100")
                                        st.markdown(f"Journal: {article['journal']}, {article['year']}")
                                        if article.get('pubmed_url'):
                                            st.markdown(f"[View on PubMed]({article['pubmed_url']})")
                        
                        # Save to history
                        source_type = "Text"
                        if use_pubmed and result.get("pubmed_articles"):
                            source_type += " with PubMed" 
                        if result.get("used_guidelines"):
                            source_type += " and Guidelines"
                        if result.get("is_triglyceride_claim"):
                            source_type += " (Triglyceride)"
                        
                        # Save with individual scores if available
                        credibility_data = result.get("individual_scores") if result.get("individual_scores") else result.get("credibility_score")
                        save_to_history(user_text, result["analysis"], credibility_data, source_type)
                    else:
                        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    with youtube_tab:
        st.write("Paste a YouTube URL to analyze the video's transcript for health misinformation.")
        
        # Add info about PubMed usage
        if use_pubmed:
            st.info("🔍 This analysis will search medical society guidelines and PubMed for evidence related to health claims in the video")
        else:
            st.info("ℹ️ Enable 'Use PubMed for evidence' in the sidebar to include scientific literature in the analysis")
        
        # YouTube URL input
        youtube_url = st.text_input("YouTube Video URL:", 
                                   placeholder="https://www.youtube.com/watch?v=... or https://youtube.com/shorts/...")
        
        # Example YouTube videos
        example_videos = {
            "None": "",
            "Mayo Clinic - Vitamin D": "https://www.youtube.com/watch?v=ZCMBjJK_xS0",
            "Cleveland Clinic - Gut Health": "https://www.youtube.com/watch?v=ww_iyhdnTPk",
            "TEDx Talk - Health": "https://www.youtube.com/watch?v=RP1AL2DU6vQ", 
            "YouTube Short": "https://youtube.com/shorts/ZD7GjNpxb_o"
        }
        
        example_video = st.selectbox("Or try an example video:", 
                                    options=list(example_videos.keys()),
                                    key="youtube_example")
        
        if example_video != "None" and example_videos[example_video]:
            youtube_url = example_videos[example_video]
            st.text(f"Using example: {youtube_url}")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            analyze_youtube_button = st.button("🎬 Analyze Video", 
                                              type="primary", 
                                              use_container_width=True)
        
        # Process YouTube analysis
        if analyze_youtube_button:
            if not openai_api_key:
                st.error("⚠️ OpenAI API Key not configured!")
                st.warning(
                    "Please enter your OpenAI API key in the sidebar to use this application."
                )
            elif not youtube_url:
                st.warning("Please enter a YouTube URL!")
            else:
                with st.spinner("Extracting video transcript and analyzing... This may take up to a minute."):
                    # Step 1: Extract video ID
                    video_id = extract_video_id(youtube_url)
                    
                    if not video_id:
                        st.error(f"Could not extract video ID from URL: {youtube_url}")
                        st.info("Please ensure you are using a valid YouTube URL format.")
                        
                        # Show supported formats
                        with st.expander("Supported YouTube URL formats"):
                            st.markdown("""
                            - Standard: `https://www.youtube.com/watch?v=VIDEO_ID`
                            - Short link: `https://youtu.be/VIDEO_ID`
                            - Shorts: `https://youtube.com/shorts/VIDEO_ID`
                            - Mobile: `https://m.youtube.com/watch?v=VIDEO_ID`
                            - Embed: `https://www.youtube.com/embed/VIDEO_ID`
                            """)
                    else:
                        # Show the extracted video ID
                        st.success(f"Detected Video ID: {video_id}")
                        
                        # Step 2: Get transcript with improved error handling
                        transcript_result = get_youtube_transcript(video_id, openai_api_key)
                        
                        if not transcript_result.get("success", False):
                            st.error(f"Could not get transcript: {transcript_result.get('error', 'Unknown error')}")
                            
                            # Check if this is a Replit environment with suggestion for manual transcript
                            if transcript_result.get("is_replit", False) and transcript_result.get("suggest_manual", False):
                                st.markdown("---")
                                st.subheader("📋 Manual Transcript Input (Replit Workaround)")
                                
                                st.info("""
                                **Since YouTube is blocking Replit, you can manually copy the transcript:**
                                
                                **Step-by-step instructions:**
                                1. 🎬 **Open your YouTube video** in a new tab
                                2. 📜 **Click the "Show transcript" button** below the video (three dots menu → "Show transcript")
                                3. 📋 **Copy all the transcript text** (you can select all with Ctrl+A/Cmd+A)
                                4. 📝 **Paste it in the text area below**
                                5. 🚀 **Click "Analyze Manual Transcript"** to process it
                                
                                **Note:** Include timestamps if you want, they'll be automatically cleaned.
                                """)
                                
                                # Large text area for manual transcript input
                                manual_transcript = st.text_area(
                                    "Paste YouTube Transcript Here:",
                                    height=250,
                                    placeholder="""Example format:
0:00 Welcome to our health discussion today
0:15 We'll be talking about nutrition and wellness
0:30 First, let's discuss the benefits of exercise...

Or just paste the plain text without timestamps.""",
                                    help="Paste the transcript you copied from YouTube. Timestamps will be automatically removed.",
                                    key="manual_transcript_input"
                                )
                                
                                # Analyze button for manual transcript
                                if st.button("🚀 Analyze Manual Transcript", type="primary", key="analyze_manual_transcript"):
                                    if not manual_transcript.strip():
                                        st.warning("Please paste a transcript in the text area above.")
                                    elif len(manual_transcript.strip()) < 50:
                                        st.warning("The transcript seems too short. Please make sure you've pasted the complete transcript.")
                                    else:
                                        with st.spinner("Analyzing manual transcript..."):
                                            # Clean the transcript (remove timestamps, extra whitespace)
                                            import re
                                            
                                            # Remove timestamp patterns like "0:00", "1:23", "12:34", etc.
                                            cleaned_transcript = re.sub(r'\b\d{1,2}:\d{2}\b', '', manual_transcript)
                                            
                                            # Remove extra whitespace and clean up
                                            cleaned_transcript = ' '.join(cleaned_transcript.split())
                                            
                                            st.success(f"✅ Processing manual transcript ({len(cleaned_transcript)} characters)")
                                            
                                            # Display the video
                                            st.subheader("Video Being Analyzed")
                                            st.video(f"https://www.youtube.com/watch?v={video_id}")
                                            
                                            # Show cleaned transcript
                                            with st.expander("View Cleaned Transcript"):
                                                st.text_area("Processed Transcript", cleaned_transcript, height=200, disabled=True)
                                            
                                            # Analyze the manual transcript
                                            if use_pubmed:
                                                selected_model = st.session_state.get('selected_model', model_option)
                                                result = analyze_youtube_transcript_with_pubmed(
                                                    cleaned_transcript, 
                                                    openai_api_key, 
                                                    model=selected_model,
                                                    ncbi_api_key=ncbi_api_key,
                                                    ncbi_email=ncbi_email
                                                )
                                            else:
                                                selected_model = st.session_state.get('selected_model', model_option)
                                                result = analyze_youtube_transcript(cleaned_transcript, openai_api_key, model=selected_model)
                                            
                                            if result["success"]:
                                                # Display results (same as automatic transcript analysis)
                                                st.subheader("Analysis Results:")
                                                
                                                # Show individual credibility scores if available
                                                if result.get("individual_scores"):
                                                    st.markdown("### Individual Claim Credibility Scores:")
                                                    
                                                    # Initialize claim analyzer for explanations
                                                    claim_analyzer = MedicalClaimAnalyzer()
                                                    
                                                    # Create a visual representation of individual scores
                                                    for claim_label, score in result["individual_scores"].items():
                                                        col1, col2 = st.columns([3, 1])
                                                        with col1:
                                                            if score >= 70:
                                                                st.success(f"**{claim_label}: {score}%** ✅")
                                                            elif score >= 40:
                                                                st.warning(f"**{claim_label}: {score}%** ⚠️")
                                                            else:
                                                                st.error(f"**{claim_label}: {score}%** ❌")
                                                        
                                                        with col2:
                                                            # Add explanation for low scores
                                                            if score < 85:
                                                                explanation = claim_analyzer.generate_score_explanation(
                                                                    f"Claim: {claim_label}", score, result, None
                                                                )
                                                                if explanation:
                                                                    st.info(f"ℹ️ {explanation}")
                                                
                                                # Display the main analysis
                                                st.markdown(result["analysis"])
                                                
                                                # Save to history
                                                save_to_history(
                                                    cleaned_transcript, 
                                                    result["analysis"], 
                                                    credibility_scores=result.get("individual_scores"),
                                                    source_type="YouTube (Manual)",
                                                    source_url=f"https://www.youtube.com/watch?v={video_id}"
                                                )
                                                
                                                st.success("✅ Analysis complete! Results saved to history.")
                                            else:
                                                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                                
                                st.markdown("---")
                            
                            # Provide helpful suggestions based on the error
                            error_msg = transcript_result.get('error', '').lower()
                            
                            if 'disabled' in error_msg or 'no transcript' in error_msg:
                                st.info("""
                                **Suggestions:**
                                1. Try a different video that has captions enabled
                                2. Look for videos from official channels (they usually have captions)
                                3. Check if the video has the 'CC' button available on YouTube
                                """)
                            elif 'unavailable' in error_msg:
                                st.info("""
                                **Suggestions:**
                                1. Check if the video is public and not age-restricted
                                2. Ensure the video hasn't been deleted or made private
                                3. Try a different video
                                """)
                            elif 'not installed' in error_msg:
                                st.info("""
                                **To fix this:**
                                1. Click 'Install/Update YouTube API' in the sidebar under Advanced Options
                                2. Restart the application after installation
                                """)
                            else:
                                # Only show general troubleshooting if not already showing Replit manual input
                                if not (transcript_result.get("is_replit", False) and transcript_result.get("suggest_manual", False)):
                                    st.info("""
                                    **General troubleshooting:**
                                    1. Try updating the YouTube API (in sidebar)
                                    2. Check your internet connection
                                    3. Try a different video
                                    4. Make sure the video has captions available
                                    """)
                            
                            # Offer alternative example videos (only if not showing manual input)
                            if not (transcript_result.get("is_replit", False) and transcript_result.get("suggest_manual", False)):
                                st.subheader("Try these videos that usually have captions:")
                                example_urls = {
                                    "Mayo Clinic - Vitamin D": "https://www.youtube.com/watch?v=ZCMBjJK_xS0",
                                    "Cleveland Clinic - Gut Health": "https://www.youtube.com/watch?v=ww_iyhdnTPk",
                                    "TED Talk - Health": "https://www.youtube.com/watch?v=8jPQjjsBbIc"
                                }
                                for name, url in example_urls.items():
                                    st.write(f"- [{name}]({url})")
                                
                        else:
                            # Success! Continue with analysis
                            try:
                                # Step 3: Analyze transcript
                                transcript = transcript_result["transcript"]
                                
                                # Show transcript metadata
                                st.info(f"Successfully retrieved transcript ({transcript_result.get('method', 'unknown method')})")
                                st.info(f"Transcript length: {len(transcript)} characters")
                                
                                # Display YouTube video
                                video_container = st.container()
                                with video_container:
                                    st.subheader("Video Being Analyzed")
                                    st.video(f"https://www.youtube.com/watch?v={video_id}")
                                
                                # Display transcript
                                with st.expander("View Transcript"):
                                    st.text_area("Video Transcript", transcript, height=200, disabled=True)
                                
                                # Analyze transcript with or without PubMed
                                if use_pubmed:
                                    # Use the enhanced version with NCBI API key if available
                                    selected_model = st.session_state.get('selected_model', model_option)
                                    result = analyze_youtube_transcript_with_pubmed(
                                        transcript, 
                                        openai_api_key, 
                                        model=selected_model,
                                        ncbi_api_key=ncbi_api_key,
                                        ncbi_email=ncbi_email
                                    )
                                else:
                                    # Use standard analysis without PubMed
                                    selected_model = st.session_state.get('selected_model', model_option)
                                    result = analyze_youtube_transcript(transcript, openai_api_key, model=selected_model)
                                
                                if result["success"]:
                                    # Display results
                                    st.subheader("Analysis Results:")
                                    
                                    # Show individual credibility scores if available
                                    if result.get("individual_scores"):
                                        st.markdown("### Individual Claim Credibility Scores:")
                                        
                                        # Initialize claim analyzer for explanations
                                        claim_analyzer = MedicalClaimAnalyzer()
                                        
                                        # Create a visual representation of individual scores
                                        for claim_label, score in result["individual_scores"].items():
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                if score >= 70:
                                                    st.success(f"**{claim_label}: {score}%** ✅")
                                                elif score >= 40:
                                                    st.warning(f"**{claim_label}: {score}%** ⚠️")
                                                else:
                                                    st.error(f"**{claim_label}: {score}%** ❌")
                                            with col2:
                                                st.progress(score / 100)
                                            
                                            # Add detailed explanation for scores below 85%
                                            if score < 85:
                                                with st.expander(f"🔍 Why did {claim_label} get {int(score)}%?", expanded=False):
                                                    try:
                                                        # Extract claim text for this specific claim
                                                        claim_number = int(claim_label.split()[-1]) - 1
                                                        claims = result.get("claims", [])
                                                        claim_text = claims[claim_number] if claim_number < len(claims) else f"Claim from YouTube video"
                                                        
                                                        # Get analysis results for this claim
                                                        analysis_results = {
                                                            'red_flags': result.get("red_flags", {}),
                                                            'evidence_quality': result.get("evidence_quality", "none"),
                                                            'guideline_alignment': result.get("guideline_alignment", "not_applicable"),
                                                            'plausibility_checks': result.get("plausibility_checks", {}),
                                                            'high_quality_evidence': result.get("high_quality_evidence", 0)
                                                        }
                                                        
                                                        # Generate detailed explanation
                                                        explanation_lines = claim_analyzer.generate_score_explanation(
                                                            claim_text, score, analysis_results, 
                                                            domain_validation=result.get("domain_validation")
                                                        )
                                                        
                                                        # Display explanation
                                                        for line in explanation_lines:
                                                            if line.strip():
                                                                st.markdown(line)
                                                            else:
                                                                st.markdown("")
                                                    except Exception as e:
                                                        st.markdown("📋 **Score factors:**")
                                                        st.markdown("• This score was calculated based on evidence quality, guideline alignment, red flags, and biological plausibility")
                                                        st.markdown("• Scores below 85% indicate areas for improvement such as adding peer-reviewed sources, explaining mechanisms, or addressing red flags")
                                                        st.markdown("• See the detailed analysis below for specific feedback")
                                    
                                    # Show overall credibility score
                                    st.markdown("### Overall Video Credibility Score:")
                                    if result.get("credibility_score") is not None:
                                        score = int(result["credibility_score"])
                                        if score >= 70:
                                            st.success(f"**{score}%** ✅")
                                        elif score >= 40:
                                            st.warning(f"**{score}%** ⚠️")
                                        else:
                                            st.error(f"**{score}%** ❌")
                                    
                                    # Add explanation about scoring
                                    with st.expander("How is the credibility score calculated?"):
                                        st.markdown("""
                                        The credibility score is based on multiple factors:
                                        
                                        **Positive factors:**
                                        - Medical society guideline support (+20 points when available)
                                        - High-quality PubMed evidence (up to +40 points)
                                        - Biological mechanism explained (+5 points)
                                        - Dosage/duration specified (+5 points)
                                        - Realistic timeline (+5 points)
                                        - Side effects acknowledged (+5 points)
                                        
                                        **Negative factors:**
                                        - Red flags like "miracle cure" claims (-10 points each, max -40)
                                        - Unrealistic timeline claims (-10 points)
                                        - Guidelines that contradict the claim (-20 points)
                                        - Known false medical claims (severe penalty, score capped at 20%)
                                        
                                        **Score ranges:**
                                        - 85-100%: Excellent credibility, strong scientific support
                                        - 70-84%: Good credibility, solid evidence base
                                        - 40-69%: Moderate credibility, mixed evidence
                                        - 0-39%: Low credibility, significant issues
                                        
                                        **Special expertise:**
                                        - Triglyceride claims validated against clinical reference ranges
                                        - Expected reduction percentages for interventions
                                        - Timeframe plausibility checks
                                        - Pharmaceutical drug name recognition and classification
                                        
                                        **Why did my claim get a lower score?**
                                        For scores below 85%, click the "🔍 Why did [Claim] get X%?" button above each individual score to see a detailed breakdown of what factors reduced the credibility and how to improve it.
                                        
                                        **Important:** The absence of guidelines doesn't penalize a claim if there's good scientific evidence from PubMed.
                                        Claims are evaluated fairly based on all available evidence. Each individual claim receives its own score, and the overall score is the average.
                                        """)
                                    
                                    # Show red flags summary if detected
                                    if result.get("red_flags"):
                                        total_flags = sum(len(flags) for flags in result["red_flags"].values())
                                        if total_flags > 0:
                                            st.warning(f"⚠️ {total_flags} red flags detected across {len(result['red_flags'])} claims")
                                    
                                    # Style the output
                                    st.markdown(result["analysis"])
                                    
                                    # Show metadata
                                    with st.expander("Analysis Metadata"):
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                                        with col2:
                                            st.metric("Model Used", result["model"])
                                        with col3:
                                            if result.get("tokens_used"):
                                                st.metric("Tokens Used", result["tokens_used"])
                                        with col4:
                                            st.metric("Claims Analyzed", len(result.get("claims", [])))
                                    
                                    # Show guidelines if available
                                    if result.get("guidelines"):
                                        with st.expander("View Relevant Guidelines"):
                                            st.write(f"Found {len(result['guidelines'])} medical society guidelines relevant to claims in this video:")
                                            
                                            for i, guideline in enumerate(result["guidelines"]):
                                                st.markdown(f"### {i+1}. {guideline['society']} ({guideline['year']})")
                                                st.markdown(f"**Category:** {guideline['category']}")
                                                st.markdown(f"**Quality Score:** {guideline.get('quality_score', 'N/A')}/100")
                                                
                                                content = guideline.get("content", "")
                                                if not content and "file_path" in guideline and os.path.exists(guideline["file_path"]):
                                                    with open(guideline["file_path"], 'r', encoding='utf-8') as f:
                                                        content = f.read()
                                                
                                                if content:
                                                    with st.expander("View Full Guideline Content"):
                                                        st.markdown(content[:2000] + "..." if len(content) > 2000 else content)
                                                        
                                                if i < len(result["guidelines"]) - 1:
                                                    st.divider()
                                    
                                    # Show PubMed articles if available
                                    if use_pubmed and result.get("pubmed_articles"):
                                        with st.expander("View PubMed Sources for Video Claims"):
                                            # Group by quality
                                            high_q = [a for a in result["pubmed_articles"] if a.get('quality_score', 0) >= 80]
                                            med_q = [a for a in result["pubmed_articles"] if 50 <= a.get('quality_score', 0) < 80]
                                            
                                            if high_q:
                                                st.markdown("### High-Quality Evidence")
                                                st.write(f"{len(high_q)} systematic reviews, meta-analyses, or clinical guidelines found")
                                                
                                                for i, article in enumerate(high_q[:5]):  # Show top 5
                                                    st.markdown(f"**{i+1}. {article['title']}**")
                                                    st.markdown(f"Quality Score: {article.get('quality_score')}/100")
                                                    st.markdown(f"Type: {', '.join(article.get('publication_types', ['Unknown']))}")
                                                    st.markdown(f"Journal: {article['journal']}, {article['year']}")
                                                    
                                                    # PubMed and DOI links
                                                    col1, col2 = st.columns(2)
                                                    with col1:
                                                        if article.get('pubmed_url'):
                                                            st.markdown(f"[View on PubMed]({article['pubmed_url']})")
                                                    with col2:
                                                        if article.get('doi_url'):
                                                            st.markdown(f"[View DOI]({article['doi_url']})")
                                                    
                                                    # Abstract in expander
                                                    with st.expander(f"View Abstract"):
                                                        st.markdown(f"{article['abstract']}")
                                                    
                                                    if i < len(high_q) - 1 and i < 4:
                                                        st.divider()
                                            
                                            if med_q:
                                                st.markdown("### Medium-Quality Evidence")
                                                st.write(f"{len(med_q)} clinical trials or observational studies found")
                                                
                                                for i, article in enumerate(med_q[:3]):  # Show top 3
                                                    st.markdown(f"**{article['title']}**")
                                                    st.markdown(f"Quality Score: {article.get('quality_score')}/100")
                                                    st.markdown(f"Journal: {article['journal']}, {article['year']}")
                                                    if article.get('pubmed_url'):
                                                        st.markdown(f"[View on PubMed]({article['pubmed_url']})")
                                    
                                    # Save to history
                                    source_type = "YouTube"
                                    if result.get("guidelines"):
                                        source_type += " with Guidelines"
                                    if use_pubmed and result.get("pubmed_articles"):
                                        source_type += " and PubMed"
                                    
                                    # Save with individual scores if available
                                    credibility_data = result.get("individual_scores") if result.get("individual_scores") else result.get("credibility_score")
                                    save_to_history(
                                        f"YouTube Transcript: {transcript_result.get('title', 'Unknown Video')}",
                                        result["analysis"],
                                        credibility_data,
                                        source_type,
                                        youtube_url
                                    )
                                else:
                                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"Error during transcript analysis: {str(e)}")

    # Disclaimer
    st.info("**Disclaimer**: This tool provides educational information only, not medical advice. Always consult healthcare professionals for personal health matters.")

with tab2:  # Ask a Question tab
    st.subheader("🤔 Ask a Health Question")
    st.write("Ask any health-related question and get evidence-based answers from medical guidelines and scientific literature.")
    
    # Question input
    user_question = st.text_area("Your health question:", 
                                placeholder="e.g., What are the symptoms of diabetes? How does exercise affect blood pressure? What foods help lower cholesterol?",
                                height=100,
                                key="health_question_input")
    
    # Example questions
    example_questions = {
        "None": "",
        "Diabetes Symptoms": "What are the early warning signs and symptoms of type 2 diabetes?",
        "Blood Pressure": "How does regular exercise help lower blood pressure?",
        "Cholesterol Diet": "What foods can help lower LDL cholesterol naturally?",
        "Sleep Quality": "What are evidence-based methods to improve sleep quality?",
        "Vitamin D": "What are the benefits and risks of vitamin D supplementation?",
        "Heart Health": "What lifestyle changes are most effective for preventing heart disease?",
        "Anxiety Management": "What non-medication approaches help manage anxiety?",
        "Immune System": "How can I naturally support my immune system?"
    }
    
    example_question = st.selectbox("Or try an example question:", 
                                   options=list(example_questions.keys()),
                                   key="question_example")
    
    if example_question != "None":
        user_question = example_questions[example_question]
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Get Answer", type="primary", use_container_width=True, key="ask_question_button")
    
    # Process question
    if ask_button and user_question:
        if not openai_api_key:
            st.error("⚠️ OpenAI API Key not configured!")
            st.warning("Please enter your OpenAI API key in the sidebar to use this application.")
        else:
            # Check for urgent medical issues
            is_urgent, urgent_message = check_medical_urgency(user_question)
            if is_urgent:
                st.error(urgent_message)
            else:
                # Detect intent
                intent = detect_query_intent(user_question, openai_api_key)
                
                if intent == "CLAIM":
                    st.info("This looks like a health claim to verify. Using claim analysis mode...")
                    # Use existing claim analysis
                    if use_pubmed:
                        selected_model = st.session_state.get('selected_model', model_option)
                        result = analyze_claim_with_pubmed(
                            user_question, 
                            openai_api_key, 
                            model=selected_model,
                            ncbi_api_key=ncbi_api_key,
                            ncbi_email=ncbi_email
                        )
                    else:
                        selected_model = st.session_state.get('selected_model', model_option)
                        result = analyze_claim_simple(user_question, openai_api_key, model=selected_model)
                else:
                    # Use Q&A mode
                    with st.spinner("Searching medical literature and formulating answer..."):
                        # Check if this is a follow-up question
                        if st.session_state.qa_history and any(word in user_question.lower() for word in ['that', 'it', 'this', 'more', 'else']):
                            result = handle_followup_question(user_question, st.session_state.qa_history, openai_api_key)
                        else:
                            result = answer_health_question(
                                user_question, 
                                openai_api_key,
                                model=model_option,
                                use_guidelines=True,
                                use_pubmed=use_pubmed,
                                ncbi_api_key=ncbi_api_key,
                                ncbi_email=ncbi_email
                            )
                        
                        if result["success"]:
                            # Display answer
                            st.markdown("### Answer:")
                            st.markdown(result["answer"])
                            
                            # Show metadata
                            with st.expander("Answer Details"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                                with col2:
                                    st.metric("Model Used", result["model"])
                                with col3:
                                    st.metric("Guidelines Used", result.get("guidelines_used", 0))
                                with col4:
                                    if result.get("tokens_used"):
                                        st.metric("Tokens Used", result["tokens_used"])
                            
                            # Show sources
                            if result.get("guidelines") or result.get("pubmed_articles"):
                                with st.expander("View Sources"):
                                    if result.get("guidelines"):
                                        st.subheader("Medical Society Guidelines")
                                        for guideline in result["guidelines"]:
                                            st.markdown(f"**{guideline['society']} ({guideline['year']})**")
                                            st.markdown(f"Category: {guideline['category']}")
                                            st.markdown(f"Quality Score: {guideline.get('quality_score', 'N/A')}/100")
                                            st.divider()
                                    
                                    if result.get("pubmed_articles"):
                                        st.subheader("Scientific Literature")
                                        high_q = [a for a in result["pubmed_articles"] if a.get('quality_score', 0) >= 80]
                                        if high_q:
                                            st.markdown("#### High-Quality Evidence")
                                            for article in high_q[:5]:
                                                st.markdown(f"**{article['title']}**")
                                                st.markdown(f"Journal: {article['journal']}, {article['year']}")
                                                if article.get('pubmed_url'):
                                                    st.markdown(f"[View on PubMed]({article['pubmed_url']})")
                                                st.divider()
                            
                            # Add to conversation history
                            st.session_state.qa_history.append({
                                'question': user_question,
                                'answer': result["answer"],
                                'timestamp': datetime.now()
                            })
                            
                            # Save to general history
                            save_to_history(
                                f"Q: {user_question}",
                                result["answer"],
                                None,  # No credibility score for Q&A
                                "Q&A" + (" with Guidelines" if result.get("guidelines") else "") + (" and PubMed" if result.get("pubmed_articles") else "")
                            )
                        else:
                            st.error(f"Failed to generate answer: {result.get('error', 'Unknown error')}")
    
    # Show conversation history
    if st.session_state.qa_history:
        st.divider()
        st.subheader("Recent Questions")
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
            with st.expander(f"Q: {qa['question'][:100]}..." if len(qa['question']) > 100 else f"Q: {qa['question']}"):
                st.markdown(f"**Asked:** {qa['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown("**Answer:**")
                st.markdown(qa['answer'])

with tab3:
    st.title("🏥 Society Guidelines Management")
    st.write("Upload and manage medical society guidelines to enhance health claim analysis. PDF files are now supported!")
    
    # Call the guidelines tab function
    add_guidelines_tab()

with tab4:
    st.subheader("Analysis History")
    
    # Load and display history
    history_df = load_history()
    
    if not history_df.empty:
        # Check if newer columns exist
        if "source_type" not in history_df.columns:
            history_df["source_type"] = "Text"
        if "source_url" not in history_df.columns:
            history_df["source_url"] = ""
        if "score_details" not in history_df.columns:
            history_df["score_details"] = "N/A"
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            source_filter = st.selectbox(
                "Filter by source:",
                ["All"] + list(history_df["source_type"].unique())
            )
        
        with col2:
            # Filter by credibility score ranges
            score_filter = st.selectbox(
                "Filter by credibility:",
                ["All", "High (70-100%)", "Medium (40-69%)", "Low (0-39%)", "Q&A (No Score)"]
            )
        
        with col3:
            # Sort options
            sort_by = st.selectbox(
                "Sort by:",
                ["Newest First", "Oldest First", "Highest Score", "Lowest Score"]
            )
        
        # Apply filters
        filtered_df = history_df.copy()
        
        if source_filter != "All":
            filtered_df = filtered_df[filtered_df["source_type"] == source_filter]
        
        if score_filter != "All":
            # Convert credibility scores to numeric, handling "N/A" values
            filtered_df["numeric_score"] = pd.to_numeric(
                filtered_df["credibility_score"].replace("N/A", None), 
                errors='coerce'
            )
            
            if score_filter == "High (70-100%)":
                filtered_df = filtered_df[filtered_df["numeric_score"] >= 70]
            elif score_filter == "Medium (40-69%)":
                filtered_df = filtered_df[(filtered_df["numeric_score"] >= 40) & (filtered_df["numeric_score"] < 70)]
            elif score_filter == "Low (0-39%)":
                filtered_df = filtered_df[filtered_df["numeric_score"] < 40]
            elif score_filter == "Q&A (No Score)":
                filtered_df = filtered_df[filtered_df["numeric_score"].isna()]
        
        # Apply sorting
        if sort_by == "Newest First":
            filtered_df = filtered_df.sort_values("timestamp", ascending=False)
        elif sort_by == "Oldest First":
            filtered_df = filtered_df.sort_values("timestamp", ascending=True)
        elif sort_by == "Highest Score":
            filtered_df["numeric_score"] = pd.to_numeric(
                filtered_df["credibility_score"].replace("N/A", -1), 
                errors='coerce'
            )
            filtered_df = filtered_df.sort_values("numeric_score", ascending=False)
        elif sort_by == "Lowest Score":
            filtered_df["numeric_score"] = pd.to_numeric(
                filtered_df["credibility_score"].replace("N/A", 999), 
                errors='coerce'
            )
            filtered_df = filtered_df.sort_values("numeric_score", ascending=True)
        
        # Display filtered history
        st.write(f"Showing {len(filtered_df)} of {len(history_df)} entries")
        
        # Display history in a table
        display_columns = ["timestamp", "query_text", "credibility_score", "source_type"]
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True
        )
        
        # Allow viewing full analysis
        if not filtered_df.empty:
            selected_entry = st.selectbox(
                "Select an entry to view full analysis:",
                options=filtered_df.index,
                format_func=lambda i: f"{filtered_df.loc[i]['timestamp']} - {filtered_df.loc[i]['query_text']}"
            )
            
            if selected_entry is not None:
                entry = filtered_df.loc[selected_entry]
                
                # Convert source_type to string if it's not already
                if not isinstance(entry["source_type"], str):
                    entry["source_type"] = str(entry["source_type"])
                
                # Display YouTube video if applicable
                if entry["source_type"].startswith("YouTube") and entry["source_url"]:
                    video_id = extract_video_id(entry["source_url"])
                    if video_id:
                        st.video(f"https://www.youtube.com/watch?v={video_id}")
                
                st.markdown("### Full Analysis")
                st.markdown(entry["full_analysis"])
                
                # Show individual scores if available
                if entry.get("score_details") and entry["score_details"] != "N/A":
                    try:
                        score_details = json.loads(entry["score_details"])
                        if isinstance(score_details, dict) and any("Claim" in k for k in score_details.keys()):
                            st.markdown("### Individual Claim Scores")
                            for claim_label, score in score_details.items():
                                if isinstance(score, (int, float)):
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        if score >= 70:
                                            st.success(f"**{claim_label}: {int(score)}%** ✅")
                                        elif score >= 40:
                                            st.warning(f"**{claim_label}: {int(score)}%** ⚠️")
                                        else:
                                            st.error(f"**{claim_label}: {int(score)}%** ❌")
                                    with col2:
                                        st.progress(score / 100)
                    except:
                        pass
                
                # Show overall credibility score with visual indicator (not for Q&A)
                if entry["credibility_score"] != "N/A" and not entry["source_type"].startswith("Q&A"):
                    try:
                        score = float(entry["credibility_score"])
                        st.markdown("### Overall Credibility Score")
                        if score >= 70:
                            st.success(f"**{int(score)}%** ✅")
                        elif score >= 40:
                            st.warning(f"**{int(score)}%** ⚠️")
                        else:
                            st.error(f"**{int(score)}%** ❌")
                    except:
                        pass
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"health_claim_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Option to clear history
            if st.button("Clear History", type="secondary"):
                if st.checkbox("Confirm deletion of all history"):
                    if os.path.exists(HISTORY_FILE):
                        os.remove(HISTORY_FILE)
                        st.success("History cleared!")
                        st.experimental_rerun()
    else:
        st.info("No analysis history yet. Start analyzing health claims to build history.")

with tab5:
    st.subheader("Common Health Misinformation Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Red Flag Patterns 🚩
        
        **Miracle Cure Claims**
        - "Cures everything"
        - "100% effective"
        - "No side effects"
        - "Instant results"
        
        **Conspiracy Language**
        - "Doctors don't want you to know"
        - "Big Pharma is hiding this"
        - "Secret ancient remedy"
        - "They're suppressing the cure"
        
        **Pseudoscience Terms**
        - "Detoxify your body"
        - "Alkaline cure"
        - "Quantum healing"
        - "Energy medicine"
        
        **Temporal Impossibility**
        - "Overnight cure"
        - "Instant weight loss"
        - "Reverse aging in days"
        - "Immediate results"
        """)
    
    with col2:
        st.markdown("""
        ### Common Misinformation Topics 📋
        
        **Alternative Medicine**
        - Unproven cancer "cures"
        - Miracle supplements
        - "Natural" always = safe fallacy
        - Energy healing claims
        
        **Diet & Nutrition**
        - Extreme detox diets
        - Single food "cures"
        - Unrealistic weight loss
        - Food combining myths
        
        **Anti-Vaccine**
        - Autism links (debunked)
        - "Natural immunity" only
        - Toxin claims
        - Conspiracy theories
        
        **Supplement Marketing**
        - Cure-all claims
        - No FDA oversight mentioned
        - Exaggerated benefits
        - Hidden risks ignored
        """)
    
    st.subheader("Understanding Credibility Scores")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🟢 High Credibility (70-100%)**
        - Supported by guidelines
        - OR high-quality evidence
        - No major red flags
        - Plausible mechanisms
        """)
    
    with col2:
        st.markdown("""
        **🟡 Medium Credibility (40-69%)**
        - Mixed evidence
        - Some red flags
        - Partial support
        - Needs more research
        """)
    
    with col3:
        st.markdown("""
        **🔴 Low Credibility (0-39%)**
        - Multiple red flags
        - Poor/no evidence
        - Implausible claims
        - Contradicts science
        """)
    
    st.info("""
    **Fair Scoring System:** Claims without medical society guidelines are NOT penalized. 
    They're evaluated based on available PubMed evidence and biological plausibility. 
    This prevents bias against newer or niche health topics that haven't been addressed by guidelines yet.
    
    **Individual Scoring:** Each claim component receives its own credibility score, allowing for nuanced evaluation of complex statements.
    
    **Semantic Search:** With Pinecone integration, the system now finds relevant guideline sections even when exact keywords don't match.
    
    **Triglyceride Expertise:** Specialized validation for triglyceride-related claims including reference range checking and intervention plausibility.
    
    **NEW: Q&A Mode:** Ask any health question and get evidence-based answers with proper citations to medical guidelines and PubMed articles.
    """)
    
    st.markdown("---")
    
    st.subheader("How to Spot Misinformation")
    
    # Create an interactive guide
    with st.expander("🔍 Quick Checklist for Evaluating Health Claims"):
        st.markdown("""
        ✅ **Check the Source**
        - Is it from a reputable medical organization?
        - Are there author credentials?
        - Is it selling something?
        
        ✅ **Look for Evidence**
        - Are studies cited?
        - Are they peer-reviewed?
        - Is the evidence cherry-picked?
        
        ✅ **Watch for Red Flags**
        - Absolute language ("always," "never")
        - Conspiracy theories
        - Miracle cure claims
        - No mention of risks
        
        ✅ **Consider Biological Plausibility**
        - Does the mechanism make sense?
        - Is it consistent with known science?
        - Are the timelines realistic?
        
        ✅ **Verify with Multiple Sources**
        - Check medical society guidelines
        - Look for systematic reviews
        - Consult healthcare providers
        """)
    
    st.subheader("Evidence Quality Hierarchy")
    
    # Visual representation of evidence hierarchy
    evidence_hierarchy = {
        "🥇 **Highest Quality**": [
            "Medical Society Guidelines (when available)",
            "Systematic Reviews & Meta-analyses",
            "Large Randomized Controlled Trials"
        ],
        "🥈 **High Quality**": [
            "Smaller RCTs",
            "Well-designed Cohort Studies",
            "Case-Control Studies"
        ],
        "🥉 **Moderate Quality**": [
            "Observational Studies",
            "Case Series",
            "Laboratory Studies"
        ],
        "⚠️ **Low Quality**": [
            "Expert Opinion",
            "Case Reports",
            "Anecdotal Evidence",
            "Social Media Posts"
        ]
    }
    
    for level, types in evidence_hierarchy.items():
        st.markdown(level)
        for evidence_type in types:
            st.markdown(f"  - {evidence_type}")
    
    st.info("""
    💡 **Pro Tip**: This tool automatically prioritizes evidence based on this hierarchy. 
    Medical society guidelines carry the most weight when available, but their absence doesn't 
    automatically make a claim less credible. High-quality PubMed evidence can strongly support 
    claims even without specific guidelines.
    """)
    
    # Triglyceride-specific section
    st.subheader("Triglyceride Reference Ranges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Normal Ranges (mg/dL)**
        - Normal: < 150
        - Borderline High: 150-199
        - High: 200-499
        - Very High: 500-999
        - Severe: ≥ 1000
        
        **Clinical Actions**
        - < 150: No intervention needed
        - 150-199: Lifestyle changes
        - 200-499: Consider medication
        - ≥ 500: Medication required
        """)
    
    with col2:
        st.markdown("""
        **Expected Reductions**
        - Low-carb diet: 20-50%
        - Fish oil: 20-30%
        - Fibrates: 30-50%
        - Weight loss: 20-30% per 10% body weight
        
        **Realistic Timeframes**
        - Diet changes: 4-12 weeks
        - Medication: 4-8 weeks
        - Exercise: 8-12 weeks
        - NOT overnight!
        """)
    
    # YouTube-specific guidance
    st.subheader("YouTube Health Content Red Flags")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Title/Thumbnail**
        - Shocking claims
        - "DOCTORS HATE THIS!"
        - Miracle cure promises
        - Before/after extremes
        """)
    
    with col2:
        st.markdown("""
        **Content Patterns**
        - No credentials shown
        - Selling products
        - Testimonials only
        - No scientific sources
        """)
    
    with col3:
        st.markdown("""
        **Claims to Avoid**
        - "Works for everyone"
        - "No diet/exercise needed"
        - "Ancient secret"
        - "Banned by FDA"
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Try These Examples
    
    Click on the appropriate tab and try these examples to see how the tool works:
    
    **'Analyze Content' Tab - Text Analysis:**
    1. **Omega-3 Fatty Acids** - A legitimate health claim with good evidence
    2. **Celery Juice** - Detox and autoimmune cure claims
    3. **Vitamin D Megadose** - Conspiracy language and unrealistic benefits
    4. **Cancer Alkaline Cure** - Dangerous misinformation about pH and cancer
    5. **Weight Loss Vinegar** - Impossible weight loss claims
    6. **High Triglycerides** - Tests the specialized triglyceride expertise
    7. **Triglyceride Denial** - Tests detection of false medical claims
    8. **Chylomicronemia False Claim** - Tests detection of specialized false claims
    
    **'Ask a Question' Tab - Q&A Mode:**
    1. **Diabetes Symptoms** - Learn about early warning signs
    2. **Blood Pressure** - Understanding exercise benefits
    3. **Cholesterol Diet** - Evidence-based dietary advice
    4. **Sleep Quality** - Science-backed sleep improvement methods
    
    The tool provides **individual credibility scores** for each claim component in verification mode, and comprehensive, cited answers in Q&A mode. Triglyceride-related claims receive specialized validation against clinical reference ranges.
    """)



# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Enhanced Health Information Assistant v4.0 | Created by a physician</p>
    <p>Now with Q&A Mode and Semantic Search via Pinecone</p>
    <p>Features Specialized Triglyceride Expertise</p>
    <p>For educational purposes only - not medical advice</p>
</div>
""", unsafe_allow_html=True)
