import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Find Your Teen\'s Secret Social Accounts - Free 60-Second Scan | Aura Balance',
  description: 'Discover hidden finsta accounts, risky connections, and digital exposure risks most parents never find. Free AI-powered scan reveals your teen\'s complete social media footprint in 60 seconds.',
  keywords: [
    'teen social media monitoring',
    'finsta finder',
    'hidden social accounts',
    'teen digital safety',
    'parental controls',
    'social media scanner',
    'teen online safety',
    'behavioral health monitoring',
    'predator protection',
    'cyberbullying prevention'
  ],
  openGraph: {
    title: 'Find Your Teen\'s Secret Social Accounts - Free Scan',
    description: 'Free 60-second scan reveals hidden accounts and digital risks most parents never discover',
    type: 'website',
    siteName: 'Aura Balance',
    images: [
      {
        url: '/social-scanner-og.jpg', // We'd need to create this
        width: 1200,
        height: 630,
        alt: 'Free Teen Social Media Scanner'
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Find Your Teen\'s Secret Social Accounts - Free Scan',
    description: 'Free 60-second scan reveals hidden accounts and digital risks',
    images: ['/social-scanner-twitter.jpg'] // We'd need to create this
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  alternates: {
    canonical: 'https://aura.com/free-teen-social-scan'
  }
};

export default function FreeScanLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      {/* Page-specific tracking scripts */}
      <script
        dangerouslySetInnerHTML={{
          __html: `
            // Enhanced tracking for landing page
            if (typeof gtag !== 'undefined') {
              gtag('event', 'page_view', {
                'page_title': 'Free Teen Social Scanner',
                'page_location': window.location.href,
                'content_group1': 'Landing Page',
                'content_group2': 'Social Scanner',
                'custom_parameter': 'free_scan_entry'
              });
            }
            
            // Facebook pixel landing page tracking
            if (typeof fbq !== 'undefined') {
              fbq('track', 'ViewContent', {
                content_name: 'Free Social Scanner',
                content_category: 'Landing Page'
              });
            }
          `
        }}
      />
      {children}
    </>
  );
}