import { Link } from "next/link";
import { Navbar, Nav } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faHandPaper } from "@fortawesome/free-solid-svg-icons";
// import { LinkContainer } from 'react-router-bootstrap';

const Layout = ({ children }) => {
  return (
    <>
      <Navbar bg="dark" variant="dark" expand="sm" sticky="top">
        <Navbar.Brand href="/" style={{ color: "#00bc8c" }}>
          <FontAwesomeIcon icon={faHandPaper} /> Handlexa
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="mr-auto">
            <Nav.Link
              href="https://sanmoydam.in/handlexa"
              style={{ color: "#00bc8c" }}
            >
              About the Project
            </Nav.Link>
            <Nav.Link
              href="https://sanmoydam.in/handlexa#two"
              style={{ color: "#00bc8c" }}
            >
              What We Do
            </Nav.Link>
            <Nav.Link
              href="https://sanmoydam.in/handlexa#one"
              style={{ color: "#00bc8c" }}
            >
              {" "}
              Technology
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Navbar>
      <div className="container">
        <div>{children}</div>
      </div>
    </>
  );
};

export default Layout;
