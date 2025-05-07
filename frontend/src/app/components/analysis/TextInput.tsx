import React from 'react';

interface TextInputProps {
  id: number;
  text: string;
  index: number;
  totalInputs: number;
  onUpdateText: (id: number, text: string) => void;
  onRemove: (id: number) => void;
}

const TextInput: React.FC<TextInputProps> = ({
  id,
  text,
  index,
  totalInputs,
  onUpdateText,
  onRemove,
}) => {
  return (
    <div className="relative flex gap-2 group">
      <div className="flex-1 relative">
        <textarea
          value={text}
          onChange={(e) => onUpdateText(id, e.target.value)}
          placeholder={`Enter text to analyze${totalInputs > 1 ? ` #${index + 1}` : ''}...`}
          className="w-full p-4 pr-10 border border-black/[.08] dark:border-white/[.12] rounded-lg h-24 bg-[#f9f9f9] dark:bg-[#161616] font-[family-name:var(--font-geist-mono)] text-sm transition-all focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 dark:focus:ring-blue-400/30 dark:focus:border-blue-400"
          required
        />
        {text && (
          <button
            type="button"
            className="absolute right-3 top-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            onClick={() => onUpdateText(id, "")}
            aria-label="Clear text"
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
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="15" y1="9" x2="9" y2="15"></line>
              <line x1="9" y1="9" x2="15" y2="15"></line>
            </svg>
          </button>
        )}
      </div>
      <button
        type="button"
        onClick={() => onRemove(id)}
        className="h-8 w-8 rounded-full flex items-center justify-center border border-black/[.08] dark:border-white/[.12] bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors self-start mt-2"
        disabled={totalInputs <= 1}
        aria-label="Remove input"
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
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
  );
};

export default TextInput;