export function absoluteUrl(path: string) {
  const base = process.env.NEXT_PUBLIC_BASE_URL
    || process.env.INTERNAL_BASE_URL
    || `http://127.0.0.1:${process.env.PORT || 3000}`
  return `${base}${path.startsWith('/') ? path : `/${path}`}`
}

