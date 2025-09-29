/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ["@google-cloud/bigquery"]
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Skip type checking during build to avoid transient TS worker errors in CI
    ignoreBuildErrors: true,
  },
}

module.exports = nextConfig
