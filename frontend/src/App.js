import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import FreshnessDetection from "./pages/FreshnessDetection";
import ProductRecognition from "./pages/ProductRecognition";
import Navbar from "./components/Navbar";
import ExpiryDateRecognition from "./pages/ExpiryDateRecognition";


function App() {
  return (
    <div className="bg-gradient-to-r from-green-500 via-teal-500 to-blue-500 font-poppins">
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/freshness" element={<FreshnessDetection />} />
        <Route path="/recognition" element={<ProductRecognition />} />
        <Route path="/expiry" element={<ExpiryDateRecognition />} />
      </Routes>
    </Router>
    </div>
  );
}

export default App;
