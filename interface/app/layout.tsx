import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Omnex',
  description: 'Everything, indexed. Nothing lost.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-omnex-bg text-omnex-text min-h-screen">
        {children}
      </body>
    </html>
  )
}
