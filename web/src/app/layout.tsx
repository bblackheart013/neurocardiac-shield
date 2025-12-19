import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'NeuroCardiac Shield — Integrated Brain-Heart Monitoring',
  description: 'Multi-modal physiological monitoring platform integrating EEG and ECG analysis for early risk detection. NYU Tandon Advanced Project.',
  authors: [
    { name: 'Mohd Sarfaraz Faiyaz' },
    { name: 'Vaibhav D. Chandgir' }
  ],
  keywords: ['EEG', 'ECG', 'physiological monitoring', 'machine learning', 'NYU', 'medical device'],
  openGraph: {
    title: 'NeuroCardiac Shield',
    description: 'Integrated Brain-Heart Monitoring System for Early Risk Detection',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        {children}
      </body>
    </html>
  )
}
