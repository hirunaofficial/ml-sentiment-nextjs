import { DetailedResult } from "../types";

// Helper function to get confidence value as percentage (0-100)
export const getConfidenceValue = (result: DetailedResult): number => {
  // Handle both formats: numeric (0-1) or string percentage
  if (result.confidence_score !== undefined) {
    return Math.min(result.confidence_score * 100, 100);
  } else if (result.confidence !== undefined) {
    // Check if confidence is a string with % sign
    if (
      typeof result.confidence === "string" &&
      result.confidence.includes("%")
    ) {
      return Math.min(parseFloat(result.confidence), 100);
    }
    // Assume it's a decimal between 0-1
    return Math.min(parseFloat(result.confidence as string) * 100, 100);
  }
  return 0;
};

// Helper function to display confidence as percentage
export const getConfidenceDisplay = (result: DetailedResult): string => {
  // Handle both formats: numeric (0-1) or string percentage
  if (result.confidence_score !== undefined) {
    return `${Math.min(result.confidence_score * 100, 100).toFixed(1)}%`;
  } else if (result.confidence !== undefined) {
    // If confidence is already a formatted string with %
    if (
      typeof result.confidence === "string" &&
      result.confidence.includes("%")
    ) {
      return result.confidence;
    }
    // Otherwise, format the number
    return `${Math.min(parseFloat(result.confidence as string) * 100, 100).toFixed(
      1
    )}%`;
  }
  return "N/A";
};