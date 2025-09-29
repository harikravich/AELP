import { withAuth } from 'next-auth/middleware'

export default withAuth({
  callbacks: {
    authorized: ({ token }) => !!token,
  },
})

const enabled = process.env.ENABLE_AUTH === '1'
export const config = {
  matcher: enabled ? [
    '/((?!api/auth|_next|static|favicon.ico|api/connections/health).*)',
  ] : [],
}
