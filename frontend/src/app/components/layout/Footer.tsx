import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="w-full max-w-4xl flex gap-5 flex-wrap items-center justify-center py-8 text-sm">
      <a
        className="flex items-center gap-2 hover:underline hover:underline-offset-4 text-gray-700 dark:text-gray-300 transition-colors hover:text-blue-600 dark:hover:text-blue-400"
        href="https://github.com/hirunaofficial/ml-sentiment-nextjs"
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
        </svg>
        View on GitHub
      </a>
      <span className="text-gray-300 dark:text-gray-600">|</span>
      <a
        className="flex items-center gap-2 hover:underline hover:underline-offset-4 text-gray-700 dark:text-gray-300 transition-colors hover:text-blue-600 dark:hover:text-blue-400"
        href="https://github.com/hirunaofficial/ml-sentiment-nextjs#readme"
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        Documentation
      </a>
      <span className="text-gray-300 dark:text-gray-600">|</span>
      <div className="text-gray-500 dark:text-gray-400 flex items-center">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="mr-1"
        >
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        Last updated: May 2025
      </div>
    </footer>
  );
};

export default Footer;