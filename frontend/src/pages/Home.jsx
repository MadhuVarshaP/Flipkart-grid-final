import React from "react";
import { Link } from "react-router-dom";
import freshness from "../images/freshness.png";
import product from "../images/product.png";
import expiry from "../images/expiry.png";

const Home = () => {
  return (
    <div className=" text-white font-poppins max-h-screen ">
      <header className="py-20 px-8">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-5xl font-bold">
            Welcome to <span className="text-blue-800">Eco Guardians</span>
          </h1>
          <p className="mt-4 text-xl italic">
            "Revolutionizing AI for a smarter, healthier tomorrow."
          </p>
          <p className="mt-6 text-lg">
            Experience state-of-the-art AI functionalities like product
            recognition and freshness detection to ensure quality and precision.
          </p>
          <div className="mt-8">
            <Link
              to="/freshness"
              className="bg-white text-green-600 hover:bg-green-100 transition px-6 py-3 rounded-lg shadow-lg mr-4 font-semibold"
            >
              Freshness Detection
            </Link>
            <Link
              to="/recognition"
              className="bg-white text-blue-600 hover:bg-blue-100 transition px-6 py-3 rounded-lg shadow-lg font-semibold"
            >
              Product Recognition
            </Link>
            <Link
              to="/expiry"
              className="bg-white text-green-600 hover:bg-green-100 transition px-6 py-3 rounded-lg shadow-lg ml-4 font-semibold"
            >
              Expiry Date Detection
            </Link>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section className="py-16 px-8 bg-white text-gray-800">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-8">
            Why Choose <span className="text-blue-500">Eco Guardians</span>?
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="group bg-gradient-to-br from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
              <img
                src={freshness} // Replace with a relevant image
                alt="Freshness Detection"
                className="w-16 h-[120px] mx-auto mb-4"
              />
              <h3 className="text-xl font-semibold text-center">
                Freshness Detection
              </h3>
              <p className="mt-2 text-sm text-center text-gray-700 group-hover:text-gray-900">
                Ensure your fruits and vegetables are at their peak freshness.
              </p>
            </div>
            {/* Feature 2 */}
            <div className="group bg-gradient-to-br from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
              <img
                src={product} // Replace with a relevant image
                alt="Product Recognition"
                className="w-16 h-[120px] mx-auto mb-4"
              />
              <h3 className="text-xl font-semibold text-center">
                Product Recognition
              </h3>
              <p className="mt-2 text-sm text-center text-gray-700 group-hover:text-gray-900">
                AI-powered recognition for accurate product identification.
              </p>
            </div>
            {/* Feature 3 */}
            <div className="group bg-gradient-to-br from-green-200 to-blue-200 p-6 rounded-lg shadow-lg hover:scale-105 transition-transform duration-300">
              <img
                src={expiry} // Replace with a relevant image
                alt="Expiry Date Detection"
                className="w-16 h-[120px] mx-auto mb-4"
              />
              <h3 className="text-xl font-semibold text-center">
                Expiry Date Detection
              </h3>
              <p className="mt-2 text-sm text-center text-gray-700 group-hover:text-gray-900">
                Never miss a date with intelligent expiry tracking.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
