'use client'

import { useState } from 'react'
import { 
  TrendingUp, 
  Brain, 
  BarChart3, 
  Activity, 
  Send,
  Loader2,
  AlertCircle,
  CheckCircle,
  XCircle,
  Target,
  DollarSign
} from 'lucide-react'

interface AnalysisResult {
  sentiment?: {
    value: string
    confidence: number
    reasoning: string
  }
  tradeSignal?: {
    signal: string
    confidence: number
    reasoning: string
    riskFactors: string[]
  }
  error?: string
}

export default function TradingLLMApp() {
  const [activeTab, setActiveTab] = useState<'sentiment' | 'signal'>('sentiment')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  
  // Sentiment Analysis State
  const [newsText, setNewsText] = useState('')
  
  // Trade Signal State
  const [symbol, setSymbol] = useState('')
  const [price, setPrice] = useState('')
  const [rsi, setRsi] = useState('')
  const [macd, setMacd] = useState('')
  const [volume, setVolume] = useState('')
  const [news, setNews] = useState('')

  const handleSentimentAnalysis = async () => {
    if (!newsText.trim()) return
    
    setLoading(true)
    setResult(null)
    
    try {
      const response = await fetch('http://localhost:8000/api/sentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newsText })
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed')
      }
      
      setResult({ sentiment: data })
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : 'Analysis failed' })
    } finally {
      setLoading(false)
    }
  }

  const handleTradeSignalGeneration = async () => {
    if (!symbol.trim() || !price.trim()) return
    
    setLoading(true)
    setResult(null)
    
    try {
      const response = await fetch('http://localhost:8000/api/trade-signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol.toUpperCase(),
          price: parseFloat(price),
          rsi: rsi ? parseFloat(rsi) : undefined,
          macd,
          volume: volume ? parseInt(volume) : undefined,
          news
        })
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || 'Signal generation failed')
      }
      
      setResult({ tradeSignal: data })
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : 'Signal generation failed' })
    } finally {
      setLoading(false)
    }
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return 'text-success-600 bg-success-50'
      case 'negative': return 'text-danger-600 bg-danger-50'
      case 'neutral': return 'text-gray-600 bg-gray-50'
      case 'mixed': return 'text-warning-600 bg-warning-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getSignalColor = (signal: string) => {
    switch (signal.toLowerCase()) {
      case 'buy': case 'strong buy': return 'text-success-600 bg-success-50'
      case 'sell': case 'strong sell': return 'text-danger-600 bg-danger-50'
      case 'hold': return 'text-warning-600 bg-warning-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 8) return 'text-success-600'
    if (confidence >= 6) return 'text-warning-600'
    return 'text-danger-600'
  }

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="max-w-6xl mx-auto mb-8">
        <div className="text-center">
          <div className="flex items-center justify-center mb-4">
            <Brain className="h-12 w-12 text-primary-600 mr-3" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-indigo-600 bg-clip-text text-transparent">
              TradingNanoLLM
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            AI-powered sentiment analysis and trade signal generation for financial markets
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-4xl mx-auto mb-8">
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
          <button
            onClick={() => setActiveTab('sentiment')}
            className={`flex-1 flex items-center justify-center py-3 px-4 rounded-md transition-all ${
              activeTab === 'sentiment'
                ? 'bg-white text-primary-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Activity className="h-5 w-5 mr-2" />
            Sentiment Analysis
          </button>
          <button
            onClick={() => setActiveTab('signal')}
            className={`flex-1 flex items-center justify-center py-3 px-4 rounded-md transition-all ${
              activeTab === 'signal'
                ? 'bg-white text-primary-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <TrendingUp className="h-5 w-5 mr-2" />
            Trade Signals
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto">
        {activeTab === 'sentiment' ? (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <Activity className="h-6 w-6 text-primary-600 mr-3" />
              <h2 className="text-2xl font-semibold">Financial Sentiment Analysis</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  News or Market Information
                </label>
                <textarea
                  value={newsText}
                  onChange={(e) => setNewsText(e.target.value)}
                  placeholder="Enter news, earnings reports, or market information to analyze..."
                  className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                />
              </div>
              
              <button
                onClick={handleSentimentAnalysis}
                disabled={loading || !newsText.trim()}
                className="w-full flex items-center justify-center py-3 px-4 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                ) : (
                  <Send className="h-5 w-5 mr-2" />
                )}
                Analyze Sentiment
              </button>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <TrendingUp className="h-6 w-6 text-primary-600 mr-3" />
              <h2 className="text-2xl font-semibold">Trade Signal Generation</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Symbol *
                </label>
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  placeholder="AAPL, TSLA, SPY..."
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Current Price *
                </label>
                <input
                  type="number"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  placeholder="175.50"
                  step="0.01"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  RSI (0-100)
                </label>
                <input
                  type="number"
                  value={rsi}
                  onChange={(e) => setRsi(e.target.value)}
                  placeholder="65"
                  min="0"
                  max="100"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  MACD Signal
                </label>
                <select
                  value={macd}
                  onChange={(e) => setMacd(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  <option value="">Select MACD Signal</option>
                  <option value="Bullish">Bullish</option>
                  <option value="Bearish">Bearish</option>
                  <option value="Neutral">Neutral</option>
                  <option value="Bullish Crossover">Bullish Crossover</option>
                  <option value="Bearish Crossover">Bearish Crossover</option>
                </select>
              </div>
              
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Volume
                </label>
                <input
                  type="number"
                  value={volume}
                  onChange={(e) => setVolume(e.target.value)}
                  placeholder="85000000"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Recent News (Optional)
                </label>
                <textarea
                  value={news}
                  onChange={(e) => setNews(e.target.value)}
                  placeholder="Latest news or announcements affecting this stock..."
                  className="w-full h-24 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                />
              </div>
            </div>
            
            <button
              onClick={handleTradeSignalGeneration}
              disabled={loading || !symbol.trim() || !price.trim()}
              className="w-full flex items-center justify-center py-3 px-4 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              ) : (
                <Target className="h-5 w-5 mr-2" />
              )}
              Generate Trade Signal
            </button>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-8 bg-white rounded-xl shadow-lg p-6 slide-in">
            {result.error ? (
              <div className="flex items-center p-4 bg-danger-50 border border-danger-200 rounded-lg">
                <XCircle className="h-5 w-5 text-danger-600 mr-3" />
                <span className="text-danger-700">{result.error}</span>
              </div>
            ) : result.sentiment ? (
              <div>
                <div className="flex items-center mb-4">
                  <CheckCircle className="h-6 w-6 text-success-600 mr-3" />
                  <h3 className="text-xl font-semibold">Sentiment Analysis Results</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <span className="font-medium">Sentiment:</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(result.sentiment.value)}`}>
                      {result.sentiment.value}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <span className="font-medium">Confidence:</span>
                    <span className={`font-semibold ${getConfidenceColor(result.sentiment.confidence)}`}>
                      {result.sentiment.confidence}/10
                    </span>
                  </div>
                  
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">Analysis Reasoning:</h4>
                    <p className="text-blue-800">{result.sentiment.reasoning}</p>
                  </div>
                </div>
              </div>
            ) : result.tradeSignal ? (
              <div>
                <div className="flex items-center mb-4">
                  <CheckCircle className="h-6 w-6 text-success-600 mr-3" />
                  <h3 className="text-xl font-semibold">Trade Signal Results</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <span className="font-medium">Signal:</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(result.tradeSignal.signal)}`}>
                      {result.tradeSignal.signal}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <span className="font-medium">Confidence:</span>
                    <span className={`font-semibold ${getConfidenceColor(result.tradeSignal.confidence)}`}>
                      {result.tradeSignal.confidence}/10
                    </span>
                  </div>
                  
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">Analysis Reasoning:</h4>
                    <p className="text-blue-800">{result.tradeSignal.reasoning}</p>
                  </div>
                  
                  {result.tradeSignal.riskFactors.length > 0 && (
                    <div className="p-4 bg-warning-50 rounded-lg">
                      <h4 className="font-medium text-warning-900 mb-2 flex items-center">
                        <AlertCircle className="h-4 w-4 mr-2" />
                        Risk Factors:
                      </h4>
                      <ul className="text-warning-800 space-y-1">
                        {result.tradeSignal.riskFactors.map((risk, index) => (
                          <li key={index} className="flex items-start">
                            <span className="text-warning-600 mr-2">•</span>
                            {risk}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="max-w-4xl mx-auto mt-12 text-center text-gray-500">
        <p className="flex items-center justify-center">
          <DollarSign className="h-4 w-4 mr-1" />
          TradingNanoLLM - AI Trading Assistant
        </p>
        <p className="text-sm mt-2">
          ⚠️ This is for educational purposes. Always do your own research before making trading decisions.
        </p>
      </div>
    </div>
  )
}
