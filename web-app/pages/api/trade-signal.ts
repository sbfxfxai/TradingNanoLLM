import { NextApiRequest, NextApiResponse } from 'next'

async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { symbol, price, rsi, macd, volume, news } = req.body

    if (!symbol || price === undefined) {
      return res.status(400).json({ error: 'Symbol and price are required' })
    }

    // Basic simulation logic (replace with actual TradingNanoLLM calls)
    let signal = 'Hold'
    let confidence = 5
    let reasoning = 'Neutral market conditions'
    let riskFactors: string[] = []

    // Simple rule-based simulation
    if (rsi && rsi < 30) {
      signal = 'Buy'
      confidence = 7
      reasoning = 'RSI indicates oversold conditions, potential for upward movement'
    } else if (rsi && rsi > 70) {
      signal = 'Sell'
      confidence = 7
      reasoning = 'RSI indicates overbought conditions, potential for downward movement'
      riskFactors.push('High RSI indicates potential reversal')
    }

    if (macd === 'Bullish' || macd === 'Bullish Crossover') {
      signal = signal === 'Sell' ? 'Hold' : 'Buy'
      confidence = Math.min(confidence + 1, 10)
      reasoning += '. MACD shows bullish momentum'
    } else if (macd === 'Bearish' || macd === 'Bearish Crossover') {
      signal = signal === 'Buy' ? 'Hold' : 'Sell'
      confidence = Math.min(confidence + 1, 10)
      reasoning += '. MACD shows bearish momentum'
      riskFactors.push('Bearish MACD indicates potential downtrend')
    }

    if (news && news.toLowerCase().includes('earnings')) {
      confidence = Math.min(confidence + 1, 10)
      reasoning += '. Earnings announcement may increase volatility'
      riskFactors.push('Earnings events increase market volatility')
    }

    const result = {
      signal,
      confidence,
      reasoning,
      riskFactors
    }

    res.status(200).json(result)
  } catch (error) {
    console.error('Trade signal error:', error)
    res.status(500).json({ error: 'Failed to generate trade signal' })
  }
}

export default handler
