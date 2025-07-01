# TradingNanoLLM Web App Deployment Guide

## Overview
This guide will help you deploy your TradingNanoLLM as a web application using:
- **Backend**: FastAPI server running your Python LLM
- **Frontend**: Next.js web application
- **Deployment**: Vercel for frontend, backend can run locally or on a cloud service

## Prerequisites
- Python 3.8+ installed
- Node.js 18+ installed
- Git installed
- Vercel account (for deployment)

## Setup Instructions

### 1. Backend Setup (FastAPI)

Navigate to the backend directory:
```bash
cd C:\trading-nano-llm\backend
```

Install Python dependencies:
```bash
python -m pip install -r requirements.txt
```

Start the FastAPI server:
```bash
python start.py
```

The server will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 2. Frontend Setup (Next.js)

Navigate to the web app directory:
```bash
cd C:\trading-nano-llm\web-app
```

Install dependencies:
```bash
npm install
```

Start the development server:
```bash
npm run dev
```

The web app will be available at: http://localhost:3000

### 3. Testing the Integration

1. Start both the backend and frontend servers
2. Open http://localhost:3000 in your browser
3. Test sentiment analysis with some financial news
4. Test trade signal generation with stock data

## Deployment Options

### Option 1: Local Development
- Run both servers locally as described above
- Use for development and testing

### Option 2: Production Deployment

#### Frontend (Vercel)
1. Push your code to a GitHub repository
2. Connect your GitHub repo to Vercel
3. Set environment variables if needed
4. Deploy

#### Backend Options

**Option A: Heroku**
1. Create a `Procfile` in the backend directory:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

2. Deploy to Heroku with the Python buildpack

**Option B: Railway**
1. Connect your GitHub repo to Railway
2. Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Option C: DigitalOcean App Platform**
1. Create a new app from your GitHub repo
2. Configure the Python service with the uvicorn command

### 4. Environment Configuration

For production, you'll need to update the API URLs in the frontend.

Create a `.env.local` file in the web-app directory:
```
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

Update the fetch URLs in `page.tsx`:
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Use in fetch calls
const response = await fetch(`${API_URL}/api/sentiment`, {
  // ...
})
```

## File Structure
```
trading-nano-llm/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Python dependencies
│   └── start.py            # Startup script
├── web-app/
│   ├── app/
│   │   ├── page.tsx        # Main web interface
│   │   ├── layout.tsx      # Layout component
│   │   └── globals.css     # Global styles
│   ├── package.json        # Node.js dependencies
│   ├── next.config.js      # Next.js configuration
│   └── tailwind.config.ts  # Tailwind CSS config
└── DEPLOYMENT.md           # This file
```

## API Endpoints

The backend provides these endpoints:

- `GET /health` - Health check
- `POST /api/sentiment` - Sentiment analysis
- `POST /api/trade-signal` - Trade signal generation
- `POST /api/portfolio-analysis` - Portfolio analysis
- `GET /api/stats` - Model statistics

## Features

### Sentiment Analysis
- Analyzes financial news and market information
- Provides sentiment classification (Positive, Negative, Neutral, Mixed)
- Includes confidence scores and reasoning

### Trade Signal Generation
- Generates Buy/Sell/Hold signals
- Considers technical indicators (RSI, MACD)
- Incorporates news sentiment
- Provides risk factor analysis

## Troubleshooting

### Common Issues

1. **Backend not starting**: Check if all Python dependencies are installed
2. **Frontend can't connect to backend**: Verify the API URL and CORS settings
3. **Model loading errors**: Ensure you have sufficient memory and the model files are accessible

### Performance Optimization

1. **Enable caching**: The backend includes prefix caching for better performance
2. **Use eager execution**: Set `enforce_eager=True` for consistent performance
3. **Monitor memory usage**: Large models require significant RAM

## Security Considerations

1. **API Rate Limiting**: Implement rate limiting for production use
2. **Input Validation**: The backend includes basic validation
3. **CORS Configuration**: Update CORS settings for your domain
4. **Environment Variables**: Keep sensitive data in environment variables

## Next Steps

1. **Custom Model Training**: Fine-tune the model on your specific trading data
2. **Real-time Data**: Integrate with financial data APIs
3. **Advanced Features**: Add portfolio optimization, backtesting, etc.
4. **Monitoring**: Set up logging and monitoring for production use

## Support

If you encounter issues:
1. Check the backend logs for errors
2. Verify all dependencies are installed
3. Ensure the model files are accessible
4. Test with the provided mock responses first
