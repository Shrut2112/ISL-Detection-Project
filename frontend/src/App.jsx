import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Webcamera from "./components/Webcam";
import Instructions from "./components/Instruction";
import SpeechToGesturesPage from "./components/SpeechToGesturesPage";
import "./App.css";

function App() {
  return (
    <Router>
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
          padding: "30px",
          background: "linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%)",
          fontFamily: "Arial, sans-serif",
        }}
      >
        <Routes>
          <Route
            path="/"
            element={
              <div className="equal-height" style={{ flexWrap: "wrap" }}>
                <div className="card">
                  <Webcamera />
                </div>
                <div className="card">
                  <Instructions />
                </div>
              </div>
            }
          />

          <Route
            path="/speech-to-gesture"
            element={<SpeechToGesturesPage />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
