import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'TradingNanoLLM - AI Trading Assistant',
  description: 'AI-powered sentiment analysis and trade signal generation for financial markets',
  keywords: 'trading, AI, LLM, sentiment analysis, trade signals, financial markets',
  authors: [{ name: 'TradingNanoLLM' }],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
        <div className="min-h-screen">
          {children}
        </div>
      </body>
    </html>
  )
}
