import { useEffect, useState } from "react";

export function useNavbarPresenter() {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { href: "#home", label: "Home" },
    { href: "#history", label: "History" },
    { href: "#about", label: "About us" },
  ];

  return { isScrolled, navLinks };
}