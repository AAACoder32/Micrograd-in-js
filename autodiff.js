/**This is javascript version of Andrej Karpathy's micrograd.
This is an autograd engine for calculating gradient of Neural Network**/

class Value {

  constructor(data, children = []) {
    this.data = data;
    this.prev = new Set(children);
    this.grad = 0.0;
    this._backward = function() {};
  }

  add(other) {
    let out = new Value(this.data + other.data, [this, other]);

    function _backward() {
      this.grad += 1 * out.grad;
      other.grad += 1 * out.grad;
    }
    out._backward = _backward;
    return out;
  }

  static add(v1, v2) {
    let out = new Value(v1.data + v2.data, [v1, v2]);

    function _backward() {
      v1.grad += 1 * out.grad;
      v2.grad += 1 * out.grad;
    }
    out._backward = _backward;

    return out;
  }

  mul(other) {
    let out = new Value(this.data * other.data, [this, other]);

    function _backward() {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    }
    out._backward = _backward;
    return out;
  }

  static mul(v1, v2) {
    let out = new Value(v1.data * v2.data, [v1, v2]);

    function _backward() {
      v1.grad += v2.data * out.grad;
      v2.grad += v1.data * out.grad;
    }
    out._backward = _backward;
    return out;
  }

  backward() {
    let topo = [];
    let visited = new Set();

    function buildTopo(v) {
      if (!visited.has(v)) {
        visited.add(v);
        for (let child of v.prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    }
    buildTopo(this);
    this.grad = 1.0;
    topo = topo.reverse();
    topo.forEach((node) => {
      node._backward();
    });
  }
}

/*** Testing ***/
/*
let a = new Value(2);
let b = new Value(-3);
let c = new Value(10);

let e = Value.mul(a, b);
let d = Value.add(e, c);
let f = new Value(-2);

let L = Value.mul(d, f);

L.backward();
console.log(L.data)
console.log(a.grad);
*/