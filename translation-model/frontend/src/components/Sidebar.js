import React from "react";

function Sidebar() {
  return (
    <div className="col-auto col-md-3 col-xl-2 px-sm-2 px-0">
      <div className="d-flex flex-column align-items-sm-start px-3 pt-2 text-white min-vh-100">
        <a
          href="/"
          className="d-flex align-items-center pb-3 mb-md-0 me-md-auto text-white text-decoration-none"
        >
          <span className="fs-5 d-none d-sm-inline">CirrusLabs</span>
        </a>
        <ul
          className="nav nav-pills flex-column mb-sm-auto mb-0 align-items-center align-items-sm-start"
          id="menu"
        >
          <li className="nav-item">
            <a href="#" className="nav-link align-middle px-0">
              <i className="fs-4 bi-house"></i>
              <span className="ms-1 d-none text-white d-sm-inline">Home</span>
            </a>
          </li>
          <li>
            <a
              href="#submenu1"
              data-bs-toggle="collapse"
              role="button"
              className="nav-link px-0 align-middle"
              aria-expanded="false"
              aria-controls="collapseModels"
            >
              <span className="ms-1 d-none text-white d-sm-inline">Models</span>
            </a>
            <ul
              className="collapse show nav flex-column ms-1"
              id="collapseModels"
              data-bs-parent="#menu"
            >
              <li className="w-100">
                <a href="#" className="nav-link px-0">
                  <span className="d-none text-white d-sm-inline">
                    Machine Translation
                  </span>
                </a>
              </li>
            </ul>
          </li>
        </ul>
        <hr />
      </div>
    </div>
  );
}

export default Sidebar;
