import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="w-full max-w-4xl pt-6">
      <div className="flex flex-col items-center text-center w-full">
        <h1 className="text-3xl sm:text-4xl font-bold mb-3 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent dark:from-blue-400 dark:to-purple-400">
          Sentiment Analysis Tool
        </h1>
        <p className="text-sm/6 text-gray-600 dark:text-gray-400 max-w-2xl">
          Analyze sentiment in text using machine learning. Add multiple
          entries for batch processing.
        </p>
      </div>
    </header>
  );
};

export default Header;