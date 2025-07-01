import { NextApiRequest, NextApiResponse } from 'next'

async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { text } = req.body

    if (!text) {
      return res.status(400).json({ error: 'Text is required' })
    }

    // Simulated analysis result (replace this with actual logic)
    const result = {
      value: 'Positive',
      confidence: 7.5,
      reasoning: 'The news indicates a strong positive outlook due to excellent revenue growth.'
    }

    // Send the analysis result
    res.status(200).json(result)
  } catch (error) {
    res.status(500).json({ error: 'Failed to analyze sentiment' })
  }
}

export default handler

