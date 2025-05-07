import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Importing custom fonts with variable names for global usage
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Metadata for the app
export const metadata: Metadata = {
  title: "Sentiment Analysis Tool",
  description: "Analyze sentiment in text using machine learning."
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en">
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="description" content={metadata.description ?? ""} />
        <meta name="application-name" content={typeof metadata.title === "string" ? metadata.title : ""} />
        <title>{typeof metadata.title === "string" ? metadata.title : "Default Title"}</title>
      </head>

      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200`}
      >
        {/* Main App Container */}
        <main className="container mx-auto p-6">
          {children}
        </main>
      </body>
    </html>
  );
}