import NextAuth from 'next-auth'
import GoogleProvider from 'next-auth/providers/google'
import Credentials from 'next-auth/providers/credentials'

export const dynamic = 'force-dynamic'

const handler = NextAuth({
  providers: [
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET
      ? [
          GoogleProvider({
            clientId: process.env.GOOGLE_CLIENT_ID as string,
            clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
            authorization: {
              params: { prompt: 'consent', access_type: 'offline', response_type: 'code' },
            },
          }),
        ]
      : []),
    // Dev fallback: simple credentials login for Pilot Mode or when Google OAuth not configured
    Credentials({
      name: 'DevLogin',
      credentials: {
        username: { label: 'Email', type: 'text' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        const user = String(credentials?.username || '').trim()
        const pass = String(credentials?.password || '').trim()
        const allowedUser = process.env.AELP2_DEV_LOGIN || 'admin@local'
        const allowedPass = process.env.AELP2_DEV_PASSWORD || 'admin'
        // Only allow credentials sign-in when PILOT_MODE=1 or explicitly enabled
        const pilot = (process.env.PILOT_MODE || '0') === '1'
        const enabled = pilot || (process.env.ENABLE_DEV_LOGIN === '1')
        if (!enabled) return null
        if (user.toLowerCase() === allowedUser.toLowerCase() && pass === allowedPass) {
          return { id: 'dev', name: 'Pilot Admin', email: user }
        }
        return null
      },
    }),
  ],
  callbacks: {
    async signIn({ account, profile }) {
      const allowedDomain = process.env.ALLOWED_EMAIL_DOMAIN
      if (allowedDomain && (profile as any)?.email) {
        return (profile as any).email.endsWith(`@${allowedDomain}`)
      }
      return true
    },
  },
  secret: process.env.NEXTAUTH_SECRET,
})

export { handler as GET, handler as POST }
