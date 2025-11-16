import { useState } from "react";
import axios from "axios";

export default function RecipeForm() {
  const [query, setQuery] = useState("");
  const [dietary, setDietary] = useState("None");
  const [health, setHealth] = useState("None");
  const [allergens, setAllergens] = useState("");
  const [calories, setCalories] = useState(0);
  const [protein, setProtein] = useState(0);
  const [fat, setFat] = useState(0);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const formatMarkdown = (text) => {
  if (!text) return text;
  
  let formatted = text;

  // ✅ Normalize bullets and numbers (forces new lines)
  formatted = formatted.replace(/(\s)\*(\s+)/g, '\n* ');
  formatted = formatted.replace(/(\s)\-(\s+)/g, '\n- ');
  formatted = formatted.replace(/(\s)(\d+)\.(\s+)/g, '\n$2. ');

  // Convert headers
  formatted = formatted.replace(/^###\s+(.+)$/gm, '<h3 class="recipe-h3">$1</h3>');
  formatted = formatted.replace(/^##\s+(.+)$/gm, '<h2 class="recipe-h2">$1</h2>');
  formatted = formatted.replace(/^#\s+(.+)$/gm, '<h1 class="recipe-h1">$1</h1>');

  // Bold & italic
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/__(.+?)__/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
  formatted = formatted.replace(/_(.+?)_/g, '<em>$1</em>');

  // ✅ Bulleted lists
  formatted = formatted.replace(/^[\*\-]\s+(.+)$/gm, '<li class="recipe-bullet">$1</li>');
  formatted = formatted.replace(/(<li class="recipe-bullet">.*?<\/li>\n?)+/g, (match) => {
    return '<ul class="recipe-list">' + match + '</ul>';
  });

  // ✅ Numbered lists (1., 2., 3.)
  formatted = formatted.replace(/^\d+\.\s+(.+)$/gm, '<li class="recipe-step">$1</li>');
  formatted = formatted.replace(/(<li class="recipe-step">.*?<\/li>\n?)+/g, (match) => {
    return '<ol class="recipe-steps">' + match + '</ol>';
  });

  return formatted;
};



  const handleSubmit = async () => {
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:5001/api/recipe", {
        query,
        dietary,
        health,
        allergens,
        calories,
        protein,
        fat,
      });
      
      // Clean the recommendation text
      const formattedResult = {
        ...response.data,
        recommendation: formatMarkdown(response.data.recommendation)
      };
      
      setResult(formattedResult);
    } catch (err) {
      console.error("Error fetching recipe:", err);
      setResult({ latency: "-", recommendation: "Failed to get recommendation. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const styles = {
    container: {
      minHeight: "100vh",
      background: "linear-gradient(135deg, #f0fdf4 0%, #ffffff 50%, #eff6ff 100%)",
      padding: "24px",
      display: "flex",
      flexDirection: "column",
      justifyContent: "flex-start",
      alignItems: "center",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
      boxSizing: "border-box",
    },
    card: {
      width: "80%",
      maxWidth: "1800px",
      margin: "0 auto",
      background: "#ffffff",
      borderRadius: "16px",
      boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
      overflow: "hidden",
      boxSizing: "border-box",
    },
    header: {
      background: "linear-gradient(135deg, #059669 0%, #10b981 100%)",
      padding: "24px 20px",
      color: "white"
    },
    title: {
      fontSize: "clamp(20px, 5vw, 28px)",
      fontWeight: "700",
      margin: "0 0 8px 0"
    },
    subtitle: {
      fontSize: "clamp(12px, 3vw, 14px)",
      opacity: "0.9",
      margin: 0
    },
    formContainer: {
      padding: "24px 20px"
    },
    inputGroup: {
      marginBottom: "24px"
    },
    label: {
      display: "block",
      fontSize: "14px",
      fontWeight: "600",
      color: "#374151",
      marginBottom: "8px"
    },
    input: {
      width: "100%",
      padding: "12px 16px",
      fontSize: "14px",
      border: "1px solid #d1d5db",
      borderRadius: "8px",
      boxSizing: "border-box",
      transition: "all 0.2s",
      outline: "none"
    },
    select: {
      width: "100%",
      padding: "12px 16px",
      fontSize: "14px",
      border: "1px solid #d1d5db",
      borderRadius: "8px",
      boxSizing: "border-box",
      transition: "all 0.2s",
      outline: "none",
      backgroundColor: "white",
      color: "#111827",
      cursor: "pointer"
    },
    gridContainer: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
      gap: "16px",
      marginBottom: "24px"
    },
    nutritionSection: {
      background: "#f9fafb",
      borderRadius: "12px",
      padding: "20px 16px",
      marginBottom: "24px"
    },
    nutritionTitle: {
      fontSize: "clamp(16px, 4vw, 18px)",
      fontWeight: "600",
      color: "#1f2937",
      marginTop: 0,
      marginBottom: "16px"
    },
    nutritionGrid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
      gap: "12px"
    },
    button: {
      width: "100%",
      padding: "14px 20px",
      fontSize: "clamp(14px, 3.5vw, 16px)",
      fontWeight: "600",
      color: "white",
      background: "linear-gradient(135deg, #059669 0%, #10b981 100%)",
      border: "none",
      borderRadius: "8px",
      cursor: "pointer",
      transition: "all 0.2s",
      boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
      opacity: loading ? 0.7 : 1,
      pointerEvents: loading ? "none" : "auto"
    },
    loaderContainer: {
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "40px 20px",
      marginTop: "24px"
    },
    spinner: {
      width: "50px",
      height: "50px",
      border: "4px solid #e5e7eb",
      borderTop: "4px solid #10b981",
      borderRadius: "50%",
      animation: "spin 1s linear infinite"
    },
    loadingText: {
      marginTop: "16px",
      color: "#6b7280",
      fontSize: "14px",
      fontWeight: "500"
    },
    resultContainer: {
      marginTop: "24px",
      background: "linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%)",
      borderRadius: "12px",
      padding: "20px 16px",
      border: "1px solid #a7f3d0"
    },
    resultHeader: {
      display: "flex",
      flexWrap: "wrap",
      justifyContent: "space-between",
      alignItems: "center",
      gap: "8px",
      marginBottom: "16px"
    },
    resultTitle: {
      fontSize: "clamp(16px, 4vw, 18px)",
      fontWeight: "600",
      color: "#1f2937",
      margin: 0
    },
    latencyBadge: {
      fontSize: "clamp(11px, 2.5vw, 12px)",
      color: "#4b5563",
      background: "white",
      padding: "6px 12px",
      borderRadius: "20px",
      whiteSpace: "nowrap"
    },
    resultText: {
      color: "#374151",
      lineHeight: "1.8",
      margin: 0,
      fontSize: "clamp(13px, 3vw, 15px)"
    },
    footer: {
      textAlign: "center",
      color: "#6b7280",
      fontSize: "clamp(11px, 2.5vw, 13px)",
      marginTop: "24px",
      padding: "0 16px"
    }
  };

  return (
    <div style={styles.container}>
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          
          .recipe-h1 {
            font-size: clamp(20px, 4vw, 24px);
            font-weight: 700;
            color: #059669;
            margin: 20px 0 12px 0;
            line-height: 1.3;
          }
          
          .recipe-h2 {
            font-size: clamp(18px, 3.5vw, 20px);
            font-weight: 700;
            color: #047857;
            margin: 16px 0 10px 0;
            line-height: 1.3;
          }
          
          .recipe-h3 {
            font-size: clamp(16px, 3vw, 18px);
            font-weight: 600;
            color: #065f46;
            margin: 14px 0 8px 0;
            line-height: 1.3;
          }
          
          .recipe-list {
            margin: 12px 0;
            padding-left: 20px;
            list-style: none;
          }
          
          .recipe-bullet {
            position: relative;
            padding-left: 20px;
            margin: 8px 0;
            color: #374151;
            line-height: 1.6;
          }
          
          .recipe-bullet:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #10b981;
            font-weight: bold;
            font-size: 18px;
          }
            .recipe-steps {
  margin: 12px 0;
  padding-left: 24px;
  list-style: decimal;
  color: #374151;
}

.recipe-step {
  margin: 8px 0;
  line-height: 1.6;
}

          
          .recipe-result strong {
            font-weight: 600;
            color: #1f2937;
          }
          
          .recipe-result em {
            font-style: italic;
            color: #4b5563;
          }
        `}
      </style>
      
      <div style={styles.card}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>NutriSense</h2>
          <p style={styles.subtitle}>Personalized meal recommendations for your health goals</p>
        </div>

        {/* Form */}
        <div style={styles.formContainer}>
          {/* Recipe Name */}
          <div style={styles.inputGroup}>
            <label style={styles.label}>Recipe or Meal Name</label>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., 'vegan curry'"
              type="text"
              style={styles.input}
              onFocus={(e) => e.target.style.borderColor = "#10b981"}
              onBlur={(e) => e.target.style.borderColor = "#d1d5db"}
              disabled={loading}
            />
          </div>

          {/* Two Column Grid */}
          <div style={styles.gridContainer}>
            {/* Dietary Preference */}
            <div>
              <label style={styles.label}>Dietary Preference</label>
              <select
                value={dietary}
                onChange={(e) => setDietary(e.target.value)}
                style={styles.select}
                onFocus={(e) => e.target.style.borderColor = "#10b981"}
                onBlur={(e) => e.target.style.borderColor = "#d1d5db"}
                disabled={loading}
              >
                <option>None</option>
                <option>Vegetarian</option>
                <option>Vegan</option>
                <option>Gluten-Free</option>
              </select>
            </div>

            {/* Health Condition */}
            <div>
              <label style={styles.label}>Health Condition / Focus</label>
              <select
                value={health}
                onChange={(e) => setHealth(e.target.value)}
                style={styles.select}
                onFocus={(e) => e.target.style.borderColor = "#10b981"}
                onBlur={(e) => e.target.style.borderColor = "#d1d5db"}
                disabled={loading}
              >
                <option>None</option>
                <option>Diabetes</option>
                <option>Hypertension</option>
                <option>Heart-Friendly</option>
                <option>Weight Loss</option>
                <option>Thyroid</option>
                <option>PCOS/PCOD</option>
                <option>Kidney-Friendly</option>
                <option>Liver Health</option>
                <option>Anemia</option>
                <option>Bone Health</option>
              </select>
            </div>
          </div>

          {/* Allergens */}
          <div style={styles.inputGroup}>
            <label style={styles.label}>Exclude Allergens</label>
            <input
              value={allergens}
              onChange={(e) => setAllergens(e.target.value)}
              placeholder="e.g., nuts, dairy"
              type="text"
              style={styles.input}
              onFocus={(e) => e.target.style.borderColor = "#10b981"}
              onBlur={(e) => e.target.style.borderColor = "#d1d5db"}
              disabled={loading}
            />
          </div>

          {/* Nutritional Targets */}
          <div style={styles.nutritionSection}>
            <h3 style={styles.nutritionTitle}>Nutritional Targets</h3>
            
            <div style={styles.nutritionGrid}>
              {/* Max Calories */}
              <div>
                <label style={styles.label}>Max Calories</label>
                <input
                  value={calories}
                  onChange={(e) => setCalories(Number(e.target.value))}
                  type="number"
                  min={0}
                  style={styles.input}
                  onFocus={(e) => e.target.style.borderColor = "#10b981"}
                  onBlur={(e) => e.target.style.borderColor = "#d1d5db"}
                  disabled={loading}
                />
              </div>

              {/* Min Protein */}
              <div>
                <label style={styles.label}>Min Protein (g)</label>
                <input
                  value={protein}
                  onChange={(e) => setProtein(Number(e.target.value))}
                  type="number"
                  min={0}
                  style={styles.input}
                  onFocus={(e) => e.target.style.borderColor = "#10b981"}
                  onBlur={(e) => e.target.style.borderColor = "#d1d5db"}
                  disabled={loading}
                />
              </div>

              {/* Max Fat */}
              <div>
                <label style={styles.label}>Max Fat (g)</label>
                <input
                  value={fat}
                  onChange={(e) => setFat(Number(e.target.value))}
                  type="number"
                  min={0}
                  style={styles.input}
                  onFocus={(e) => e.target.style.borderColor = "#10b981"}
                  onBlur={(e) => e.target.style.borderColor = "#d1d5db"}
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            style={styles.button}
            onMouseEnter={(e) => {
              if (!loading) {
                e.target.style.transform = "scale(1.02)";
                e.target.style.boxShadow = "0 10px 15px -3px rgba(0, 0, 0, 0.1)";
              }
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = "scale(1)";
              e.target.style.boxShadow = "0 4px 6px -1px rgba(0, 0, 0, 0.1)";
            }}
            disabled={loading}
          >
            {loading ? "Finding Your Perfect Recipe..." : "Get Personalized Recommendation"}
          </button>

          {/* Loading Indicator */}
          {loading && (
            <div style={styles.loaderContainer}>
              <div style={styles.spinner}></div>
              <p style={styles.loadingText}>Analyzing your preferences and finding the best match...</p>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <div style={styles.resultContainer}>
              <div style={styles.resultHeader}>
                <h3 style={styles.resultTitle}>Your Recommendation</h3>
                <span style={styles.latencyBadge}>⚡ {result.latency}s</span>
              </div>
              <div 
                className="recipe-result"
                style={styles.resultText}
                dangerouslySetInnerHTML={{ __html: result.recommendation }}
              />
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <p style={styles.footer}>
        Powered by NutriSense AI • Personalized nutrition recommendations
      </p>
    </div>
  );
}