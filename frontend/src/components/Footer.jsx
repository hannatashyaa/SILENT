import React from 'react'
import { Link } from 'react-router-dom'
import { Facebook, Twitter, Instagram } from 'lucide-react'
import logo from '../assets/logo.png'

const Footer = () => {
  return (
    <footer className="bg-blue-gradient text-white">
      <div className="container mx-auto px-4 py-12">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="md:col-span-1">
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-10 h-10 rounded-lg flex items-center justify-center">
                <img src={logo} alt="SILENT Logo" className="w-12 h-12 object-contain rounded-lg p-1" />
              </div>
              <span className="font-bold text-xl">SILENT</span>
            </div>
            <p className="text-blue-100 text-sm mb-4">
              Sign language interpretation<br />
              and expression translator.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold text-lg mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-blue-100 hover:text-white transition-colors">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/translate" className="text-blue-100 hover:text-white transition-colors">
                  Translate
                </Link>
              </li>
              <li>
                <Link to="/history" className="text-blue-100 hover:text-white transition-colors">
                  History
                </Link>
              </li>
            </ul>
          </div>

          {/* Our Services */}
          <div>
            <h3 className="font-semibold text-lg mb-4">Our Services</h3>
            <ul className="space-y-2 text-blue-100 text-sm">
              <li>Real-time Sign Translation</li>
              <li>Image-Based Translation</li>
              <li>Camera-Based Translation</li>
              <li>API Integration for Accessibility</li>
            </ul>
          </div>

          {/* Community & Support */}
          <div>
            <h3 className="font-semibold text-lg mb-4">Community & Support</h3>
            <ul className="space-y-2 text-blue-100 text-sm">
              <li>Community Forum</li>
              <li>Contribute Data</li>
              <li>Research Collaboration</li>
              <li>Report an Issue</li>
            </ul>
          </div>
        </div>

        {/* Social Media & Copyright */}
        <div className="border-t border-blue-600 mt-8 pt-8">
          <div className="flex flex-col md:flex-row items-center justify-between">
            {/* Social Media Icons */}
            <div className="flex items-center space-x-4 mb-4 md:mb-0">
              <a
                href="#"
                className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center hover:bg-blue-700 transition-colors"
              >
                <Facebook size={20} />
              </a>
              <a
                href="#"
                className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center hover:bg-blue-700 transition-colors"
              >
                <Twitter size={20} />
              </a>
              <a
                href="#"
                className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center hover:bg-blue-700 transition-colors"
              >
                <Instagram size={20} />
              </a>
              {/* <a
                href="#"
                className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center hover:bg-blue-700 transition-colors"
              >
                <span className="text-lg">🎵</span>
              </a> */}
            </div>

            {/* Copyright */}
            <div className="text-blue-100 text-sm text-center md:text-right">
              © 2025 BISINDO & SIBI Translator.
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer