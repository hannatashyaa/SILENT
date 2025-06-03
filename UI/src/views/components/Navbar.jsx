import logo from "../../assets/logo.png";
import { useNavbarPresenter } from "../../presenters/NavBarPresenter";

export default function Navbar() {
  const { isScrolled, navLinks } = useNavbarPresenter();

  return (
    <nav
      className={`fixed w-full top-0 z-50 h-16 px-6 flex justify-between items-center transition mx-auto md:mx-0 ${
        isScrolled ? "bg-[#009DFFCE]/80 backdrop-blur-md" : "bg-[#009DFF]"
      } text-white`}
    >
      <img src={logo} alt="SILENT Logo" className="h-10" />
      <ul className="flex space-x-6">
        {navLinks.map((link) => (
          <li key={link.href}>
            <a href={link.href} className="hover:font-semibold">
              {link.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
