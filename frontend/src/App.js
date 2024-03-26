import React, { useState } from "react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import ModelsView from "./components/ModelsView";
import { AppProvider } from "./context/AppContext";

import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  return (
    <div className="app">
      <main>
        <div className="container-fluid">
          <Row className="flex-nowrap">
            <Col
              className="col-auto col-md-3 col-xl-2 px-sm-2 px-0"
              style={{ backgroundColor: "#4C97D5" }}
            >
              <div className="container-fluid">
                <Row className="row flex-nowrap">
                  <Sidebar />
                </Row>
              </div>
            </Col>
            <Col>
              <Container fluid className="px-0">
                <Row>
                  <Col style={{ backgroundColor: "#86B2D6" }}>
                    <Dashboard />
                  </Col>
                </Row>
                <Row>
                  <Col style={{ backgroundColor: "white" }}>
                    <ModelsView />
                  </Col>
                </Row>
              </Container>
            </Col>
          </Row>
        </div>
      </main>
    </div>
  );
}

export default App;
