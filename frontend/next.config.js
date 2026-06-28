//frontend/next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
  httpAgentOptions: {
    keepAlive: true,
  },
};

module.exports = nextConfig;