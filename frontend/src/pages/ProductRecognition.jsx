import React, { useState } from "react";

const ProductRecognition = () => {
  const [folderFiles, setFolderFiles] = useState([]);
  const [result, setResult] = useState(null);

  const handleFolderUpload = e => {
    const files = Array.from(e.target.files);
    setFolderFiles(files);
  };

  const handleSubmit = async () => {
    if (folderFiles.length === 0) {
      alert("Please upload a folder with images.");
      return;
    }

    const formData = new FormData();
    folderFiles.forEach(file => {
      formData.append("images", file);
    });

    try {
      const response = await fetch("http://localhost:5000/predict-folder", {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      setResult(data); // Set the result from backend
    } catch (error) {
      console.error("Error:", error);
      alert("Prediction failed.");
    }
  };

  const handleDownloadExcel = async () => {
    try {
      const response = await fetch("http://localhost:5000/download-excel", {
        method: "GET"
      });

      if (!response.ok) {
        throw new Error("Failed to download the Excel file");
      }

      // Create a blob from the response
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      // Create a link element to trigger the download
      const a = document.createElement("a");
      a.href = url;
      a.download = "product_predictions.xlsx";
      document.body.appendChild(a);
      a.click();
      a.remove(); // Clean up the link element
    } catch (error) {
      console.error("Error downloading Excel:", error);
      alert("Failed to download the Excel file.");
    }
  };

  return (
    <div className="text-white min-h-screen py-10 font-poppins">
      <div className="max-w-4xl mx-auto text-center">
        {/* Header Section */}
        <h1 className="text-4xl font-bold mb-4 text-blue-800">
          Product Recognition
        </h1>
        <p className="text-lg italic mb-6">
          "Upload a folder containing product images, and let AI predict their
          brand."
        </p>

        {/* Folder Upload Section */}
        <div className="mt-6 bg-white text-gray-800 rounded-lg p-6 shadow-md">
          <label
            htmlFor="folder-upload"
            className="block text-lg font-medium mb-4"
          >
            Upload a Folder
          </label>
          <input
            type="file"
            webkitdirectory="true"
            directory="true"
            multiple
            onChange={handleFolderUpload}
            className="block w-full border border-gray-300 rounded-md p-2 cursor-pointer file:bg-green-500 file:border-0 file:text-white file:py-2 file:px-4 file:rounded-md file:shadow-md"
          />
          {folderFiles.length > 0 &&
            <p className="mt-4 text-sm text-gray-700">
              {folderFiles.length} files selected.
            </p>}
        </div>

        {/* Submit Button */}
        <div className="mt-6">
          <button
            onClick={handleSubmit}
            className="bg-blue-500 text-white px-6 py-3 rounded-lg shadow-md font-semibold hover:bg-blue-600 transition"
          >
            Submit for Detection
          </button>
        </div>

        {/* Results Section */}
        {result &&
          <div className="mt-10 p-6 bg-white text-gray-800 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">Detection Results</h2>
            <ul className="space-y-4">
              {result.map((item, index) =>
                <li
                  key={index}
                  className="bg-gray-100 p-4 rounded-md shadow hover:bg-gray-200 transition"
                >
                  <h3 className="text-xl font-bold text-green-500">
                    {item.predicted_brand}
                  </h3>
                </li>
              )}
            </ul>
          </div>}
        {/* Download Excel Button */}
        <div className="mt-6">
          <button
            onClick={handleDownloadExcel}
            className="bg-green-500 text-white px-6 py-3 rounded-lg shadow-md font-semibold hover:bg-green-600 transition"
          >
            Download Excel file
          </button>
        </div>
      </div>

      {/* Features Section */}
      <section className="mt-16 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Feature 1 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Bulk Recognition
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Upload entire folders of images to detect product brands in one
              go.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              High Accuracy
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Leverage AI-powered models to accurately classify products.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Easy File Management
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Drag-and-drop folders for quick and seamless uploads.
            </p>
          </div>

          {/* Feature 4 */}
          <div className="group bg-gradient-to-tr from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Detailed Analysis
            </h3>
            <p className="text-gray-600 group-hover:text-gray-900">
              Gain valuable insights with comprehensive product descriptions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ProductRecognition;
