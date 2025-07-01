import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List
import asyncio
import json

# Import your trading LLM components
try:
    from trading_nanovllm import TradingLLM, SamplingParams
    from trading_nanovllm.trading_utils import analyze_sentiment, generate_trade_signal
    from trading_nanovllm.trading_utils import SentimentAnalyzer, TradeSignalGenerator
    TRADING_LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TradingNanoLLM: {e}")
    print("Using mock responses for development")
    TRADING_LLM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TradingNanoLLM API",
    description="AI-powered sentiment analysis and trade signal generation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for LLM
llm = None
sentiment_analyzer = None
signal_generator = None
sampling_params = None

# Request models
class SentimentRequest(BaseModel):
    text: str

class TradeSignalRequest(BaseModel):
    symbol: str
    price: float
    rsi: Optional[float] = None
    macd: Optional[str] = None
    volume: Optional[int] = None
    news: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# Response models
class SentimentResponse(BaseModel):
    value: str
    confidence: float
    reasoning: str

class TradeSignalResponse(BaseModel):
    signal: str
    confidence: float
    reasoning: str
    riskFactors: List[str]

# Initialize the LLM on startup
@app.on_event("startup")
async def startup_event():
    global llm, sentiment_analyzer, signal_generator, sampling_params
    
    if TRADING_LLM_AVAILABLE:
        try:
            logger.info("Initializing TradingNanoLLM...")
            
            # Initialize with optimizations
            llm = TradingLLM(
                "Qwen/Qwen2-0.5B-Instruct", 
                enforce_eager=True,
                enable_prefix_caching=True
            )
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=0.6, 
                max_tokens=256,
                do_sample=True
            )
            
            # Initialize analyzers
            sentiment_analyzer = SentimentAnalyzer(llm)
            signal_generator = TradeSignalGenerator(llm)
            
            logger.info("TradingNanoLLM initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize TradingNanoLLM: {e}")
            logger.info("Continuing with mock responses...")
    else:
        logger.info("TradingNanoLLM not available, using mock responses")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=llm is not None,
        version="1.0.0"
    )

# Sentiment analysis endpoint
@app.post("/api/sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(request: SentimentRequest):
    try:
        logger.info(f"Analyzing sentiment for text: {request.text[:100]}...")
        
        if llm and sentiment_analyzer and TRADING_LLM_AVAILABLE:
            # Use real TradingNanoLLM
            try:
                # Basic sentiment analysis
                sentiment = analyze_sentiment(request.text, llm, sampling_params)
                
                # Enhanced analysis with confidence
                result = sentiment_analyzer.analyze_with_confidence(
                    request.text, 
                    num_samples=3
                )
                
                return SentimentResponse(
                    value=result['sentiment'],
                    confidence=result['confidence'],
                    reasoning=f"Analysis based on {len(result['samples'])} samples. {sentiment}"
                )
                
            except Exception as e:
                logger.error(f"Error in LLM analysis: {e}")
                # Fallback to mock
                return get_mock_sentiment(request.text)
        else:
            # Use mock response
            return get_mock_sentiment(request.text)
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze sentiment")

# Trade signal generation endpoint
@app.post("/api/trade-signal", response_model=TradeSignalResponse)
async def generate_trade_signal_endpoint(request: TradeSignalRequest):
    try:
        logger.info(f"Generating trade signal for {request.symbol} at ${request.price}")
        
        if llm and signal_generator and TRADING_LLM_AVAILABLE:
            # Use real TradingNanoLLM
            try:
                # Create market data dictionary
                market_data = {
                    "symbol": request.symbol,
                    "price": request.price,
                    "rsi": request.rsi,
                    "macd": request.macd,
                    "volume": request.volume,
                    "news": request.news or "",
                }
                
                # Generate detailed analysis
                analysis = signal_generator.generate_with_analysis(market_data)
                
                return TradeSignalResponse(
                    signal=analysis['signal'],
                    confidence=analysis['confidence'],
                    reasoning=analysis['reasoning'],
                    riskFactors=analysis.get('risk_factors', [])
                )
                
            except Exception as e:
                logger.error(f"Error in LLM signal generation: {e}")
                # Fallback to mock
                return get_mock_trade_signal(request)
        else:
            # Use mock response
            return get_mock_trade_signal(request)
            
    except Exception as e:
        logger.error(f"Error in trade signal generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate trade signal")

# Portfolio analysis endpoint
@app.post("/api/portfolio-analysis")
async def analyze_portfolio(portfolio_data: dict):
    try:
        if llm and signal_generator and TRADING_LLM_AVAILABLE:
            try:
                signals = signal_generator.generate_portfolio_signals(portfolio_data)
                return signals
            except Exception as e:
                logger.error(f"Error in portfolio analysis: {e}")
                return {"error": "Portfolio analysis failed"}
        else:
            return {"message": "Portfolio analysis not available in mock mode"}
            
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze portfolio")

# Mock functions for development/fallback
def get_mock_sentiment(text: str) -> SentimentResponse:
    """Generate mock sentiment analysis"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['profit', 'gain', 'up', 'beat', 'positive', 'growth']):
        return SentimentResponse(
            value="Positive",
            confidence=7.5,
            reasoning="Text contains positive financial indicators such as profit, growth, or beating expectations."
        )
    elif any(word in text_lower for word in ['loss', 'down', 'decline', 'negative', 'miss']):
        return SentimentResponse(
            value="Negative",
            confidence=7.0,
            reasoning="Text contains negative financial indicators such as losses, declines, or missing expectations."
        )
    elif any(word in text_lower for word in ['mixed', 'uncertain', 'volatile']):
        return SentimentResponse(
            value="Mixed",
            confidence=6.0,
            reasoning="Text shows mixed signals with both positive and negative indicators."
        )
    else:
        return SentimentResponse(
            value="Neutral",
            confidence=5.5,
            reasoning="Text shows neutral sentiment without strong positive or negative indicators."
        )

def get_mock_trade_signal(request: TradeSignalRequest) -> TradeSignalResponse:
    """Generate mock trade signal"""
    signal = "Hold"
    confidence = 5.0
    reasoning = "Neutral market conditions"
    risk_factors = []
    
    # RSI-based logic
    if request.rsi:
        if request.rsi < 30:
            signal = "Buy"
            confidence = 7.5
            reasoning = "RSI indicates oversold conditions, potential for upward movement"
        elif request.rsi > 70:
            signal = "Sell"
            confidence = 7.0
            reasoning = "RSI indicates overbought conditions, potential for downward movement"
            risk_factors.append("High RSI indicates potential reversal")
    
    # MACD logic
    if request.macd:
        if request.macd in ["Bullish", "Bullish Crossover"]:
            if signal != "Sell":
                signal = "Buy"
            confidence = min(confidence + 1.0, 10.0)
            reasoning += ". MACD shows bullish momentum"
        elif request.macd in ["Bearish", "Bearish Crossover"]:
            if signal != "Buy":
                signal = "Sell"
            confidence = min(confidence + 1.0, 10.0)
            reasoning += ". MACD shows bearish momentum"
            risk_factors.append("Bearish MACD indicates potential downtrend")
    
    # News sentiment
    if request.news:
        news_lower = request.news.lower()
        if any(word in news_lower for word in ['earnings', 'report', 'announcement']):
            confidence = min(confidence + 0.5, 10.0)
            reasoning += ". Recent news may impact price volatility"
            risk_factors.append("News events increase market volatility")
        
        if any(word in news_lower for word in ['positive', 'beat', 'exceeds']):
            if signal == "Sell":
                signal = "Hold"
            elif signal == "Hold":
                signal = "Buy"
            confidence = min(confidence + 1.0, 10.0)
            reasoning += ". Positive news sentiment"
    
    return TradeSignalResponse(
        signal=signal,
        confidence=round(confidence, 1),
        reasoning=reasoning,
        riskFactors=risk_factors
    )

# Statistics endpoint
@app.get("/api/stats")
async def get_stats():
    if llm and TRADING_LLM_AVAILABLE:
        try:
            stats = llm.get_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": "Stats not available"}
    else:
        return {"message": "Stats not available in mock mode"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
