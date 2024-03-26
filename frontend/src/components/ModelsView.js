import React from "react";
import Card from "react-bootstrap/Card";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";

function ModelsView() {
  return (
    <Row className="mx-auto">
      <Col>
        <Card className="mx-auto" style={{ width: "18rem" }}>
          <Card.Img
            variant="top"
            src="https://dummyimage.com/900x400/b0b0b0/000000.jpg"
          />
          <Card.Body>
            <Card.Title>Google T5</Card.Title>
            <Card.Text>
              Model from Google. Capable of translating English, French,
              Romanian, and German.
            </Card.Text>
          </Card.Body>
        </Card>
      </Col>
      <Col>
        <Card className="mx-auto" style={{ width: "18rem" }}>
          <Card.Img
            variant="top"
            src="https://dummyimage.com/900x400/b0b0b0/000000.jpg"
          />
          <Card.Body>
            <Card.Title>Facebook NLLB</Card.Title>
            <Card.Text>
              Model from Facebook. Capable of translating over 200 languages.
            </Card.Text>
          </Card.Body>
        </Card>
      </Col>
      <Col>
        <Card className="mx-auto" style={{ width: "18rem" }}>
          <Card.Img
            variant="top"
            src="https://dummyimage.com/900x400/b0b0b0/000000.jpg"
          />
          <Card.Body>
            <Card.Title>Helsinki Opus</Card.Title>
            <Card.Text>
              Model from Helsinki University. Finetuned for only English to
              French translation.
            </Card.Text>
          </Card.Body>
        </Card>
      </Col>
    </Row>
  );
}

export default ModelsView;
