import os
import io
import base64
import asyncio
import httpx
import fitz
import time
from PIL import Image
from typing import List, Optional, Union, Tuple
from dotenv import load_dotenv
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.logger import get_logger
from config import config

load_dotenv()

logger = get_logger("ImageOCR")


class ImageOCR:
    """
    Image OCR service for extracting text from images and PDFs using USF Vision API.
    Converts PDF pages to images and uses vision model for OCR.
    
    OPTIMIZED for high-volume extraction (400-500 pages in 1-2 minutes):
    - Connection pooling with persistent httpx client
    - Semaphore-based concurrency control
    - JPEG compression for smaller payloads
    - Lower DPI for faster processing
    - Parallel PDF page conversion
    """
    
    def __init__(self, api_key: str = None, batch_size: int = None):
        self.api_key = api_key or config.USF_API_KEY
        self.api_url = config.USF_API_URL
        self.model = config.OCR_MODEL
        self.batch_size = batch_size or config.OCR_BATCH_SIZE
        self.dpi = config.OCR_DPI
        self.max_tokens = config.OCR_MAX_TOKENS
        self.temperature = config.OCR_TEMPERATURE
        self.timeout = config.OCR_TIMEOUT
        self.max_concurrent = getattr(config, 'OCR_MAX_CONCURRENT', 100)
        self.image_quality = getattr(config, 'OCR_IMAGE_QUALITY', 70)
        self.data_extracted_folder = getattr(config, 'DATA_EXTRACTED_FOLDER', 'data_extracted')
        
        # Persistent HTTP client for connection pooling
        self._client: Optional[httpx.AsyncClient] = None
        # Semaphore for concurrency control
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info(f"ImageOCR service initialized (batch_size={self.batch_size}, "
                   f"max_concurrent={self.max_concurrent}, dpi={self.dpi}, quality={self.image_quality})")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            # Configure for high concurrency
            limits = httpx.Limits(
                max_keepalive_connections=100,
                max_connections=200,
                keepalive_expiry=30.0
            )
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(self.timeout), connect=30.0),
                limits=limits,
                http2=True  # HTTP/2 for multiplexing
            )
        return self._client
    
    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _image_to_base64_optimized(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string with JPEG compression for speed."""
        buffer = io.BytesIO()
        # Use JPEG for smaller size and faster transfer
        # Convert to RGB if necessary (JPEG doesn't support alpha)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=self.image_quality, optimize=True)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string (legacy method)."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL Images using PyMuPDF (optimized)."""
        images = []
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"📄 Converting {total_pages} PDF pages to images (DPI={self.dpi})...")
            
            for page_num in range(total_pages):
                page = doc[page_num]
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
                # Progress logging every 50 pages
                if (page_num + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (page_num + 1) / elapsed
                    logger.info(f"   Converted {page_num + 1}/{total_pages} pages ({rate:.1f} pages/sec)")
            
            doc.close()
            elapsed = time.time() - start_time
            logger.info(f"✅ PDF conversion complete: {len(images)} pages in {elapsed:.1f}s "
                       f"({len(images)/elapsed:.1f} pages/sec)")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise Exception(f"Error converting PDF to images: {str(e)}")
        return images
    
    async def extract_text_from_image(self, image: Union[str, Image.Image], use_optimized: bool = True) -> str:
        """
        Extract text from a single image using USF Vision API.
        Uses connection pooling and semaphore for high concurrency.
        
        Args:
            image: Either a file path or PIL Image object
            use_optimized: Use JPEG compression for faster transfer
        
        Returns:
            Extracted text from the image
        """
        semaphore = await self._get_semaphore()
        
        async with semaphore:
            try:
                if isinstance(image, str):
                    img = Image.open(image)
                else:
                    img = image
                
                # Use optimized JPEG encoding for speed
                if use_optimized:
                    base64_image = self._image_to_base64_optimized(img)
                    image_type = "image/jpeg"
                else:
                    base64_image = self._image_to_base64(img)
                    image_type = "image/png"
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract ALL text from this image. Provide the complete text content exactly as it appears, preserving the structure and formatting. If this is a legal document, ensure all clauses, terms, and conditions are captured accurately."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{image_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": self.temperature,
                    "stream": False,
                    "max_tokens": self.max_tokens
                }
                
                # Use persistent client with connection pooling
                client = await self._get_client()
                response = await client.post(self.api_url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"USF API error: {response.text}")
                    raise Exception(f"USF API error: {response.text}")
                
                result = response.json()
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return text
                    
            except Exception as e:
                logger.error(f"Error extracting text from image: {e}")
                raise Exception(f"Error extracting text from image: {str(e)}")
    
    def extract_text_from_image_sync(self, image: Union[str, Image.Image]) -> str:
        """Synchronous version of extract_text_from_image."""
        import asyncio
        return asyncio.run(self.extract_text_from_image(image))
    
    async def _extract_single_page(self, page_data: Tuple[int, Image.Image]) -> Tuple[int, str]:
        """
        Extract text from a single page image.
        
        Args:
            page_data: Tuple of (page_number, image)
        
        Returns:
            Tuple of (page_number, extracted_text)
        """
        page_num, img = page_data
        try:
            text = await self.extract_text_from_image(img)
            return (page_num, text)
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num}: {e}")
            return (page_num, f"[Error extracting text: {str(e)}]")
    
    def _save_extracted_text(self, text: str, original_filename: str) -> str:
        """
        Save extracted text to the data_extracted folder.
        
        Args:
            text: Extracted text content
            original_filename: Original PDF filename
            
        Returns:
            Path to saved file
        """
        # Create data_extracted folder if it doesn't exist
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        output_folder = os.path.join(project_root, self.data_extracted_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Create output filename (replace .pdf with .txt)
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_{timestamp}.txt"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"💾 Saved extracted text to: {output_path}")
        return output_path
    
    async def extract_text_from_pdf(self, pdf_path: str, save_to_file: bool = True) -> str:
        """
        Extract text from a PDF by converting pages to images and using OCR.
        
        OPTIMIZED for high-volume extraction (400-500 pages in 1-2 minutes):
        - All pages processed in parallel (no batching delays)
        - Connection pooling for reduced overhead
        - Semaphore controls max concurrent requests
        - JPEG compression for smaller payloads
        
        Args:
            pdf_path: Path to the PDF file
            save_to_file: Whether to save extracted text to data_extracted folder
        
        Returns:
            Extracted text from all pages
        """
        start_time = time.time()
        logger.info(f"🚀 Starting OPTIMIZED OCR extraction from PDF: {pdf_path}")
        logger.info(f"   Max concurrent requests: {self.max_concurrent}")
        logger.info(f"   DPI: {self.dpi}, JPEG quality: {self.image_quality}")
        
        # Step 1: Convert PDF to images
        images = self._pdf_to_images(pdf_path)
        total_pages = len(images)
        conversion_time = time.time() - start_time
        
        # Step 2: Prepare all page tasks
        page_data = [(i + 1, img) for i, img in enumerate(images)]
        
        # Step 3: Process ALL pages in parallel (semaphore controls concurrency)
        logger.info(f"🔄 Starting parallel OCR for {total_pages} pages...")
        ocr_start = time.time()
        
        # Create all tasks at once - semaphore will control actual concurrency
        tasks = [self._extract_single_page(pd) for pd in page_data]
        
        # Progress tracking with asyncio.as_completed
        results = {}
        completed = 0
        last_log_time = time.time()
        
        for coro in asyncio.as_completed(tasks):
            page_num, text = await coro
            results[page_num] = text
            completed += 1
            
            # Log progress every 2 seconds or every 50 pages
            current_time = time.time()
            if current_time - last_log_time >= 2.0 or completed % 50 == 0:
                elapsed = current_time - ocr_start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_pages - completed) / rate if rate > 0 else 0
                logger.info(f"   📊 Progress: {completed}/{total_pages} pages "
                           f"({rate:.1f} pages/sec, ETA: {eta:.0f}s)")
                last_log_time = current_time
        
        ocr_time = time.time() - ocr_start
        
        # Step 4: Combine results in order
        all_text = []
        for page_num in sorted(results.keys()):
            all_text.append(f"--- Page {page_num} ---\n{results[page_num]}")
        
        full_text = "\n\n".join(all_text)
        total_time = time.time() - start_time
        
        # Log performance summary
        logger.info(f"")
        logger.info(f"✅ OCR EXTRACTION COMPLETE")
        logger.info(f"   📄 Total pages: {total_pages}")
        logger.info(f"   📝 Total characters: {len(full_text):,}")
        logger.info(f"   ⏱️  PDF conversion: {conversion_time:.1f}s")
        logger.info(f"   ⏱️  OCR processing: {ocr_time:.1f}s ({total_pages/ocr_time:.1f} pages/sec)")
        logger.info(f"   ⏱️  Total time: {total_time:.1f}s ({total_pages/total_time:.1f} pages/sec)")
        
        # Step 5: Save to file if requested
        if save_to_file:
            saved_path = self._save_extracted_text(full_text, pdf_path)
            logger.info(f"   💾 Saved to: {saved_path}")
        
        return full_text
    
    def extract_text_from_pdf_sync(self, pdf_path: str) -> str:
        """Synchronous version of extract_text_from_pdf."""
        import asyncio
        return asyncio.run(self.extract_text_from_pdf(pdf_path))


image_ocr = ImageOCR()


async def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convenience function to extract text from PDF using OCR.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Extracted text
    """
    return await image_ocr.extract_text_from_pdf(pdf_path)


def extract_text_from_pdf_sync(pdf_path: str) -> str:
    """
    Synchronous convenience function to extract text from PDF using OCR.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Extracted text
    """
    return image_ocr.extract_text_from_pdf_sync(pdf_path)
