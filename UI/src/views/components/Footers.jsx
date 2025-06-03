import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faFacebookF,
  faYoutube,
  faInstagram,
  faTiktok,
} from "@fortawesome/free-brands-svg-icons";

export default function Footer() {
  return (
    <footer className="bg-[#009DFF] text-white pt-10 pb-6" id="about">
      <div className="max-w-6xl mx-auto px-4 grid grid-cols-1 md:grid-cols-4 gap-6">
        <div>
          <h2 className="text-2xl font-bold">SILENT</h2>
          <p className="mt-2 text-sm">
            sign language interpretation and expression translator.
          </p>
        </div>

        <div>
          <h3 className="font-semibold mb-2">Quick Links</h3>
          <ul className="text-sm space-y-1">
            <li><a href="#" className="hover:underline">Home</a></li>
            <li><a href="#" className="hover:underline">Translate</a></li>
            <li><a href="#" className="hover:underline">History</a></li>
          </ul>
        </div>

        <div>
          <h3 className="font-semibold mb-2">Our Services</h3>
          <ul className="text-sm space-y-1">
            <li>Real-time Sign Translation</li>
            <li>Image-Based Translation</li>
            <li>Camera-Based Translation</li>
            <li>API Integration for Accessibility</li>
          </ul>
        </div>

        <div>
          <h3 className="font-semibold mb-2">Community & Support</h3>
          <ul className="text-sm space-y-1">
            <li>Community Forum</li>
            <li>Contribute Data</li>
            <li>Research Collaboration</li>
            <li>Report an Issue</li>
          </ul>
        </div>
      </div>

      <div className="mt-10 border-t border-blue-500 pt-6 text-center text-sm">
        <div className="flex justify-center space-x-4 mt-4">
          <a
            href="#"
            className="w-10 h-10 flex items-center justify-center rounded-full text-white border-2 border-white hover:bg-white hover:text-[#009DFF] transition"
          >
            <FontAwesomeIcon icon={faFacebookF} />
          </a>
          <a
            href="#"
            className="w-10 h-10 flex items-center justify-center rounded-full text-white border-2 border-white hover:bg-white hover:text-[#009DFF] transition"
          >
            <FontAwesomeIcon icon={faYoutube} />
          </a>
          <a
            href="#"
            className="w-10 h-10 flex items-center justify-center rounded-full text-white border-2 border-white hover:bg-white hover:text-[#009DFF] transition"
          >
            <FontAwesomeIcon icon={faInstagram} />
          </a>
          <a
            href="#"
            className="w-10 h-10 flex items-center justify-center rounded-full text-white border-2 border-white hover:bg-white hover:text-[#009DFF] transition"
          >
            <FontAwesomeIcon icon={faTiktok} />
          </a>
        </div>
        <p className="mt-4">© 2025 BISINDO & SIBI Translator.</p>
      </div>
    </footer>
  );
}