# Document Parser API

## Overview

The Document Parser API is a Flask-based REST service that uses advanced OCR (Optical Character Recognition) techniques to extract structured data from certificate documents. It supports multiple image enhancement strategies, intelligent pattern matching, and exports data in various formats.

## Key Features

- **Multi-strategy OCR processing** with different Tesseract configurations
- **Image enhancement techniques** to improve OCR accuracy
- **Intelligent pattern matching** for certificate data extraction
- **Batch processing** of multiple documents
- **Multiple export formats** (JSON, CSV, Excel)
- **Comprehensive error handling**

## Installation

### Prerequisites

- Python 3.7+
- Tesseract OCR engine
- System dependencies for OpenCV and Pillow

### Dependencies

```bash
pip install pillow pytesseract opencv-python pandas flask flask-cors
```

### Tesseract Installation

- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## API Endpoints

### Health Check
- **Endpoint**: `GET /api/health`
- **Description**: Verify API status
- **Response**: JSON with status and timestamp

### Parse Single Document
- **Endpoint**: `POST /api/parse`
- **Description**: Process a single certificate document
- **Parameters**: Multipart form with `file` field
- **Response**: Extracted certificate data

### Parse Batch Documents
- **Endpoint**: `POST /api/parse/batch`
- **Description**: Process multiple documents (max 10)
- **Parameters**: Multipart form with `files[]` field
- **Response**: Array of parsing results

### Export Data
- **Endpoint**: `POST /api/export/<format>`
- **Description**: Export parsed data in specified format
- **Formats**: `json`, `csv`, `excel`
- **Parameters**: JSON data in request body
- **Response**: File download

### OCR Text Extraction
- **Endpoint**: `POST /api/ocr-text`
- **Description**: Extract raw OCR text without parsing
- **Parameters**: Multipart form with `file` field
- **Response**: Extracted text

### Supported Formats
- **Endpoint**: `GET /api/supported-formats`
- **Description**: List supported file formats
- **Response**: JSON with format information

### Test Parameters
- **Endpoint**: `GET /api/test-parameters`
- **Description**: Get known test parameters and specifications
- **Response**: JSON with parameter details

## Usage Examples

### Parse a Single Document

```bash
curl -X POST -F "file=@certificate.jpg" http://localhost:5000/api/parse
```

### Batch Processing

```bash
curl -X POST -F "files[]=@cert1.jpg" -F "files[]=@cert2.jpg" http://localhost:5000/api/parse/batch
```

### Export Data

```bash
curl -X POST -H "Content-Type: application/json" -d '{"parsed_data": {...}}' http://localhost:5000/api/export/csv
```

## Data Structure

### CertificateData Class

The extracted data follows this structure:

```python
{
    "certificate_number": str,
    "product": str,
    "product_number": str,
    "customer_order_number": str,
    "customer_order_number_2": str,
    "batch_number": str,
    "manufacturing_date": str,
    "expiry_date": str,
    "specification": str,
    "version": str,
    "test_results": {
        "summary": dict,
        "detailed_table": list,
        "headers": list
    }
}
```

### Test Parameters

The API recognizes these test parameters:

1. **Refractive Index** (Brechungsindex)
   - Range: 1.3660 - 1.3720
   - Unit: keine
   - Method: 100004300

2. **Density** (Dichte)
   - Range: 1.040 - 1.070
   - Unit: g/ml
   - Method: 100000400

3. **MIT**
   - Range: 8.50 - 9.50
   - Unit: %
   - Method: 100009900

4. **BIT**
   - Range: 4.60 - 5.20
   - Unit: %
   - Method: 100009700

5. **Appearance** (Äußeres)
   - Expected: "Entspricht"
   - Unit: keine
   - Method: 100011800

## Configuration

### Flask App Settings

- **Max file size**: 16MB
- **Upload folder**: System temp directory
- **Allowed extensions**: png, jpg, jpeg, tiff, bmp, pdf

### Environment Variables

Set these before running the application:

```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

## Running the Application

### Development Mode

```bash
python app.py --host localhost --port 5000 --debug
```

### Production Deployment

For production use, deploy with a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Error Handling

The API provides comprehensive error handling for:

- File size limits (413)
- Invalid file types (400)
- Parsing errors (500)
- Missing files (400)

All errors return JSON responses with details.

## AdvancedDocumentParser Class

The core parsing functionality includes:

1. **Image Enhancement**:
   - Contrast adjustment
   - Sharpening
   - Grayscale conversion
   - CLAHE equalization
   - Edge enhancement

2. **OCR Strategies**:
   - High DPI processing
   - Dense text parsing
   - Sparse text parsing
   - Single block processing
   - Word-based extraction

3. **Pattern Matching**:
   - Certificate numbers
   - Product information
   - Batch numbers
   - Dates
   - Test results

## Limitations

- Best results with high-quality images (300+ DPI)
- German language documents work best
- Complex layouts may reduce accuracy
- Handwritten text is not supported

## Troubleshooting

### Common Issues

1. **Tesseract not found**:
   - Install Tesseract and ensure it's in PATH
   - Set `pytesseract.pytesseract.tesseract_cmd` if needed

2. **Image processing errors**:
   - Check OpenCV and Pillow installations

3. **Memory errors**:
   - Reduce batch size or image resolution

### Logging

The application logs to stdout with INFO level. Set `logging.basicConfig(level=logging.DEBUG)` for more detailed logs.

## Support

For issues and questions, please check:
- Tesseract OCR documentation
- OpenCV image processing guides
- Flask framework documentation

## License

This project is provided as-is without warranty. Please ensure proper licensing for Tesseract and other dependencies in production environments.