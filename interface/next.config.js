/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: '127.0.0.1', port: '8000' },
      { protocol: 'http', hostname: 'localhost',  port: '8000' },
      { protocol: 'http', hostname: 'api',        port: '8000' },
    ],
  },
}

module.exports = nextConfig
