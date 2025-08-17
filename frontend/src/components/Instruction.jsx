import React from "react";
import gestures from "../assets/ndian-sign-language-for-numbers-and-alphabets.png";
import "./Instruction.css";
import { Link } from "react-router-dom";

const Instructions = () => {
  return (
    <div className="instructions-card">
      <h2 className="main-title">✋ Gesture Instructions</h2>

      <div className="image-section">
        <img src={gestures} alt="gesture photo" className="gesture-image" />
        <p className="image-caption">📚 Indian Sign Language</p>
      </div>

      <div className="controls-section">
        <div className="control-item delete-control">
          <div className="control-icon">🗑️</div>
          <div className="control-text">
            <h3>Delete Letter</h3>
            <p>Swipe left palm → right</p>
          </div>
          <div className="gesture-demo">
            <span className="arrow">👈</span>
            <span className="arrow">👉</span>
          </div>
        </div>

        <div className="control-item undo-control">
          <div className="control-icon">↩️</div>
          <div className="control-text">
            <h3>Undo Delete</h3>
            <p>Swipe right palm → left</p>
          </div>
          <div className="gesture-demo">
            <span className="arrow">👉</span>
            <span className="arrow">👈</span>
          </div>
        </div>
      </div>

      <div className="button-wrapper">
        <Link to='/speech-to-gesture'><button className="main-button" >🎤 Speech-to-Gestures</button></Link>
      </div>
    </div>
  );
};

export default Instructions;
