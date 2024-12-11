import React, { useState } from "react";

const ExpiryDateRecognition = () => {
  const [frontImage, setFrontImage] = useState(null);
  const [backImage, setBackImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageUpload = (e, type) => {
    const file = e.target.files[0];
    if (type === "front") {
      setFrontImage(file);
    } else if (type === "back") {
      setBackImage(file);
    }
  };

  const handleSubmit = async () => {
    if (!frontImage || !backImage) {
      alert("Please upload both front and back images.");
      return;
    }

    const formData = new FormData();
    formData.append("front_image", frontImage);
    formData.append("back_image", backImage);

    try {
      const response = await fetch(
        "http://localhost:5000/recognize-expiry-date",
        {
          method: "POST",
          body: formData
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Something went wrong");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to recognize expiry date. " + error.message);
    }
  };
  const handleDownloadExcel = async () => {
    try {
      const response = await fetch("http://localhost:5000/download-excel", {
        method: "GET"
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to download the file.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "Expiry_Brand_Details.xlsx"; // Specify file name
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading file:", error);
      alert("Failed to download the file. " + error.message);
    }
  };

  return (
    <div className="text-white min-h-screen py-10 font-poppins">
      <div className="max-w-4xl mx-auto text-center">
        {/* Header Section */}
        <h1 className="text-4xl font-bold mb-4 text-blue-800">
          Expiry Date Recognition
        </h1>
        <p className="text-lg italic mb-6">
          "Upload front and back images of the product to detect its expiry
          date."
        </p>

        {/* Image Upload Section */}
        <div className="mt-6 bg-white text-gray-800 rounded-lg p-6 shadow-md">
          <label
            htmlFor="front-upload"
            className="block text-lg font-medium mb-4"
          >
            Upload Front Image
          </label>
          <input
            type="file"
            onChange={e => handleImageUpload(e, "front")}
            className="block w-full border border-gray-300 rounded-md p-2 cursor-pointer file:bg-green-500 file:border-0 file:text-white file:py-2 file:px-4 file:rounded-md file:shadow-md mb-6"
          />
          {frontImage &&
            <p className="text-sm text-gray-700">
              Selected File: {frontImage.name}
            </p>}

          <label
            htmlFor="back-upload"
            className="block text-lg font-medium mt-6 mb-4"
          >
            Upload Back Image
          </label>
          <input
            type="file"
            onChange={e => handleImageUpload(e, "back")}
            className="block w-full border border-gray-300 rounded-md p-2 cursor-pointer file:bg-green-500 file:border-0 file:text-white file:py-2 file:px-4 file:rounded-md file:shadow-md"
          />
          {backImage &&
            <p className="text-sm text-gray-700">
              Selected File: {backImage.name}
            </p>}
        </div>

        {/* Submit Button */}
        <div className="mt-6">
          <button
            onClick={handleSubmit}
            className="bg-green-500 text-white px-6 py-3 rounded-lg shadow-md font-semibold hover:bg-green-600 transition"
          >
            Submit for Expiry Date Detection
          </button>
        </div>

        {/* Results Section */}
        {result &&
          <div className="mt-10 p-6 bg-white text-gray-800 rounded-lg shadow-md max-w-3xl mx-auto">
            <h2 className="text-2xl font-semibold mb-4">Detection Results</h2>
            <div className="space-y-4">
              <div className="bg-gray-100 p-4 rounded-md shadow hover:bg-gray-200 transition">
                <h3 className="text-xl font-bold text-green-500">
                  Brand Name:
                </h3>
                <p className="text-lg">
                  {result.brand_name || "Not Available"}
                </p>
              </div>
              <div className="bg-gray-100 p-4 rounded-md shadow hover:bg-gray-200 transition">
                <h3 className="text-xl font-bold text-green-500">
                  Detected Expiry Date:
                </h3>
                <p className="text-lg">
                  {result.expiry_date || "Not Available"}
                </p>
              </div>
              <div className="bg-gray-100 p-4 rounded-md shadow hover:bg-gray-200 transition">
                <h3 className="text-xl font-bold text-green-500">
                  Expiration Status:
                </h3>
                <p className="text-lg">
                  {result.status_message || "Not Available"}
                </p>
              </div>
            </div>
          </div>}
        <button
          onClick={handleDownloadExcel}
          className="mt-6 bg-blue-500 text-white px-6 py-3 rounded-lg shadow-md font-semibold hover:bg-blue-600 transition"
        >
          Download Excel File
        </button>
      </div>
      {/* Features Section */}
      <section className="mt-16 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Feature 1 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Automated Detection
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Quickly identify expiry dates using advanced image processing.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              High Accuracy
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Ensure precise detection of expiry dates for better inventory
              management.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              User-Friendly Interface
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Simple and intuitive interface for easy image uploads and results.
            </p>
          </div>

          {/* Feature 4 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Real-Time Feedback
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Get instant feedback on the expiry status of your products.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ExpiryDateRecognition;
