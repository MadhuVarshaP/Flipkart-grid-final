import React, { useState } from "react";

const FreshnessDetection = () => {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleImageUpload = e => {
    const file = e.target.files[0];
    setImage(file);
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("image", image);
    try {
      const response = await fetch("http://localhost:5000/detect-freshness", {
        method: "POST",
        body: formData,
      });
    
    
      if (!response.ok) {
        console.error("Response not OK", response.statusText);
        throw new Error("Failed to process the image.");
      }
    
      const data = await response.json();
      console.log("Backend Response:", data); // Debug response
      if (data.detections) {
        setResult(data.detections);
      } else {
        console.error("No detections in response:", data);
        alert("No detections found in the response.");
      }
    } catch (error) {
      console.error("Error in handleSubmit:", error);
      alert("Failed to process the image. Please try again.");
    } finally {
      setLoading(false);
    }
  }    

  return (
    <div className="text-white min-h-screen py-10 font-poppins">
      <div className="max-w-4xl mx-auto text-center">
        {/* Header Section */}
        <h1 className="text-4xl font-bold mb-4 text-blue-800">
          Freshness Detection
        </h1>
        <p className="text-lg italic mb-6">
          "Ensure the quality of your real-world objects with our AI-powered
          freshness detection tool."
        </p>

        {/* Instructions Section */}
        <p className="text-lg mb-4">
          Upload an image of your real-world object to analyze its freshness.
        </p>

        {/* Image Upload Section */}
        <div className="mt-6 bg-white text-gray-800 rounded-lg p-6 shadow-md">
          <label
            htmlFor="image-upload"
            className="block text-lg font-medium mb-4"
          >
            Upload an Image
          </label>
          <input
            type="file"
            onChange={handleImageUpload}
            className="block w-full border border-gray-300 rounded-md p-2 cursor-pointer file:bg-green-500 file:border-0 file:text-white file:py-2 file:px-4 file:rounded-md file:shadow-md"
          />
          {image &&
            <p className="text-sm text-gray-700 mt-4">
              Selected File: {image.name}
            </p>}
        </div>

        {/* Submit Button */}
        <div className="mt-6">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className={`px-6 py-3 rounded-lg shadow-lg font-semibold transition ${loading
              ? "bg-gray-400 text-gray-800 cursor-not-allowed"
              : "bg-green-500 text-white hover:bg-green-600"}`}
          >
            {loading ? "Processing..." : "Analyze Freshness"}
          </button>
        </div>

        {/* Result Display */}
        {result &&
          <div className="mt-10 p-6 bg-white text-gray-800 rounded-lg shadow-md max-w-3xl mx-auto">
            <h2 className="text-2xl font-semibold mb-2">Detection Results</h2>
            <ul>
              {result.map((detection, index) =>
                <li key={index} className="text-lg">
                  <strong>{detection.product}</strong> ({detection.freshness}) -
                  Confidence: {detection.confidence.toFixed(2)}
                </li>
              )}
            </ul>
          </div>}
      </div>

      {/* Features Section */}
      <section className="mt-16 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Feature 1 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Real-Time Detection
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Analyze object quality instantly with cutting-edge AI models.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              User-Friendly
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Easy-to-use interface designed for everyone, from farmers to
              consumers.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Accurate Insights
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Gain reliable insights into the freshness and quality of your
              produce.
            </p>
          </div>

          {/* Feature 4 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Save Resources
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Reduce waste by detecting freshness and making informed decisions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default FreshnessDetection;
