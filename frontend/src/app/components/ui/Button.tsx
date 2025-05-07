import React from 'react';

type ButtonVariant = 'primary' | 'secondary';
type ButtonSize = 'default' | 'small';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
  icon?: React.ReactNode;
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'secondary',
  size = 'default',
  isLoading = false,
  icon,
  children,
  className,
  ...props
}) => {
  const baseClasses = "rounded-lg border border-solid transition-colors flex items-center justify-center font-medium text-sm";
  
  const variantClasses = {
    primary: "border-transparent bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 dark:from-blue-500 dark:to-purple-500 dark:hover:from-blue-600 dark:hover:to-purple-600 text-white shadow-sm",
    secondary: "border-black/[.08] dark:border-white/[.12] hover:bg-gray-50 dark:hover:bg-gray-800"
  };
  
  const sizeClasses = {
    default: "h-10 px-4",
    small: "h-8 px-3 text-xs"
  };
  
  const disabledClasses = "disabled:opacity-50 disabled:cursor-not-allowed";
  
  const buttonClasses = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${disabledClasses} ${className || ''}`;

  return (
    <button 
      className={buttonClasses} 
      disabled={isLoading || props.disabled}
      {...props}
    >
      {isLoading ? (
        <>
          <svg
            className="animate-spin -ml-1 mr-2 h-4 w-4 text-current"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          {typeof children === 'string' ? 'Loading...' : children}
        </>
      ) : (
        <>
          {icon && <span className="mr-2">{icon}</span>}
          {children}
        </>
      )}
    </button>
  );
};

export default Button;