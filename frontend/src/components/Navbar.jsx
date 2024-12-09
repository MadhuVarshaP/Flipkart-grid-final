import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className=" text-white shadow-lg p-5 font-poppins ">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center space-x-2">
          <span className="text-2xl font-bold tracking-wide">
            Eco Guardians
          </span>
        </Link>

        <div className="hidden md:flex space-x-6">
          <Link
            to="/freshness"
            className="text-lg font-medium transition duration-300 hover:text-blue-800 hover:underline"
          >
            Freshness Detection
          </Link>
          <Link
            to="/recognition"
            className="text-lg font-medium transition duration-300 hover:text-blue-800 hover:underline"
          >
            Product Recognition
          </Link>
          <Link
            to="/expiry"
            className="text-lg font-medium transition duration-300 hover:text-blue-800 hover:underline"
          >
            Expiry Date Recognition
          </Link>
        </div>
      </div>

      {/* Dropdown Menu for Mobile */}
      <div className="md:hidden bg-gray-900 px-4 py-2">
        <Link
          to="/freshness"
          className="block py-2 text-white hover:text-blue-300"
        >
          Freshness Detection
        </Link>
        <Link
          to="/recognition"
          className="block py-2 text-white hover:text-blue-300"
        >
          Product Recognition
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
