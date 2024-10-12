use compute::gcd;

#[derive(Debug, Clone)]
enum Token {
    /// pure integer number
    Int(i64),
    /// integer part, digital part
    Double(i64, i64),
    /// upper, lower
    Fractor(i64, i64),
    /// a var
    X,
    /// compare (not exist ><=)
    C,
    Plus,
    Minus,
    Multiply,
    Divide,
    Percent,
    Equal,
    Left,
    Right,
    GreatThan,
    LessThan,
}

mod compute {
    pub type Num = (i64, i64);
    pub type Pair = (Num, Num, i64);
    pub fn number_len(mut x: i64) -> i64 {
        let mut len = 0;
        while x != 0 {
            x /= 10;
            len += 1;
        }
        len
    }
    pub fn e10(mut i: i64) -> i64 {
        let mut res = 1;
        while i > 0 {
            res *= 10;
            i -= 1;
        }
        res
    }
    pub fn double_to_frac((x, y): Num) -> Num {
        let exp = e10(number_len(y));
        if x < 0 {
            simplify_fractor(x * exp - y, exp)
        } else {
            simplify_fractor(x * exp + y, exp)
        }
    }

    pub fn align(mut x: i64, mut y: i64) -> (i64, i64) {
        let mut i = x;
        let mut j = y;
        while i != 0 || j != 0 {
            i /= 10;
            j /= 10;
            if i == 0 {
                x *= 10;
            }
            if j == 0 {
                y *= 10;
            }
        }
        (x, y)
    }

    pub fn simplify((x, y): Num, p: i64) -> Num {
        match p {
            1 => (x, 0),
            2 => (x, cancel_zeros(y)),
            3 => simplify_fractor(x, y),
            _ => (0, 0),
        }
    }
    pub fn simplify_fractor(x: i64, y: i64) -> Num {
        let g = gcd(x.abs(), y.abs());
        (x / g, y / g)
    }
    pub fn cancel_zeros(mut x: i64) -> i64 {
        while x % 10 == 0 {
            x /= 10;
        }
        x
    }
    pub fn gcd(x: i64, y: i64) -> i64 {
        if y == 0 {
            x
        } else {
            gcd(y, x % y)
        }
    }

    pub fn plus(l: Num, r: Num) -> Num {
        let g = gcd(l.1, r.1);
        let x = l.0 * (r.1 / g) + r.0 * (l.1 / g);
        let y = l.1 * (r.1 / g);
        (x, y)
    }
    pub fn minus(l: Num, r: Num) -> Num {
        let g = gcd(l.1, r.1);
        let x = l.0 * (r.1 / g) - r.0 * (l.1 / g);
        let y = l.1 * (r.1 / g);
        (x, y)
    }
    pub fn times(l: Num, r: Num) -> Num {
        let x = l.0 * r.0;
        let y = l.1 * r.1;
        let neg = x.is_negative() ^ y.is_negative();
        let x = x.abs();
        let y = y.abs();
        let g = gcd(x, y);
        if neg {
            (-x / g, y / g)
        } else {
            (x / g, y / g)
        }
    }
    pub fn divide(l: Num, r: Num) -> Num {
        let x = l.0 * r.1;
        let y = l.1 * r.0;
        let neg = x.is_negative() ^ y.is_negative();
        let x = x.abs();
        let y = y.abs();
        let g = gcd(x, y);
        if neg {
            (-x / g, y / g)
        } else {
            (x / g, y / g)
        }
    }

    pub fn xEy(x: i64, y: i64) -> String {
        if y < 10 {
            return x.to_string();
        }
        let mut res = String::new();
        if x.is_negative() {
            res += "-";
        }
        if x < y {
            res += "0.";
            let mut y = y / 10;
            while x < y {
                res += "0";
                y /= 10;
            }
            res += &x.abs().to_string();
        } else {
            let x = x.abs();
            res += &(x / y).to_string();
            res += ".";
            res += &(x % y).to_string();
        }
        res
    }
}

impl Token {
    fn pirority(&self) -> i64 {
        match self {
            Token::C => 0,
            Token::Equal | Token::GreatThan | Token::LessThan => 0,
            Token::Plus | Token::Minus => 1,
            Token::Divide | Token::Multiply => 2,
            Token::Percent => 3,
            Token::Left | Token::Right => 4,
            Token::X => 6,
            Token::Int(_) => 6,
            Token::Double(_, _) => 6,
            Token::Fractor(_, _) => 6,
        }
    }
    const NULL: Self = Token::Int(0);

    // Frac > Double > Int
    fn get_num_pirority(&self) -> i64 {
        match self {
            Token::Int(..) => 1,
            Token::Double(..) => 2,
            Token::Fractor(..) => 3,
            _ => 0,
        }
    }

    fn get_num(&self) -> (i64, i64) {
        match self {
            Token::Int(x) => (*x, 1),                               // x_1
            Token::Double(x, y) | Token::Fractor(x, y) => (*x, *y), // x_y
            _ => (0, 0),
        }
    }

    fn from_num_at_p((x, y): (i64, i64), p: i64) -> Self {
        match p {
            1 if x % y == 0 => Token::Int(x / y),
            1 | 3 => Token::Fractor(x, y),
            2 => {
                let mut a = x;
                let mut b = y;
                let mut i = b;
                while i > 1 {
                    if i % 10 == 0 {
                        i /= 10;
                    }
                    if i % 2 == 0 {
                        i /= 2;
                        a *= 5;
                        b *= 5;
                    } else if i % 5 == 0 {
                        i /= 5;
                        a *= 2;
                        b *= 2;
                    } else {
                        return Token::Fractor(x, y);
                    }
                }
                Token::Double(a, b)
            }
            _ => Token::NULL,
        }
    }
    fn from_num((x, y): (i64, i64)) -> Self {
        if x % y == 0 {
            Token::Int(x / y)
        } else {
            let g = gcd(x.abs(), y.abs());
            Self::from_num_at_p((x / g, y / g), 2)
        }
    }

    fn cmp(lhs: compute::Num, rhs: compute::Num) -> Self {
        let c = lhs.0 * rhs.1 - lhs.1 * rhs.0;
        if c > 0 {
            Token::GreatThan
        } else if c == 0 {
            Token::Equal
        } else {
            Token::LessThan
        }
    }
}

impl Default for Token {
    fn default() -> Self {
        Self::NULL
    }
}
impl Default for &Token {
    fn default() -> Self {
        &Token::NULL
    }
}

#[derive(Debug, Clone)]
struct Parser {
    pub token: Vec<Token>,
    pub error: bool,
}

impl Parser {
    pub fn new(s: String) -> Self {
        let mut token = Vec::new();
        /* Virtual machine state:
         * 0 New
         * 0-9 0    => 1 // Digit
         * 0-9 1    => 1 // Digit
         * 0-9 2    => 2 // Number
         * .   0,1  => 2 // Number
         * +-%)1,2  => 0
         * (   1,2,3=> 0
         * x        => 3 // X
         */
        let mut state = 0;
        let mut digit = 0;
        let mut frac = 0;
        let mut error = false;
        for c in s.chars() {
            while match c {
                '0'..='9' if state == 0 => {
                    digit = c as i64 - '0' as i64;
                    state = 1;
                    false
                }
                '0'..='9' if state == 1 => {
                    digit = digit * 10 + (c as i64 - '0' as i64);
                    false
                }
                '0'..='9' if state == 2 => {
                    digit = digit * 10 + (c as i64 - '0' as i64);
                    frac = frac * 10;
                    false
                }
                '.' if state == 0 || state == 1 => {
                    state = 2;
                    frac = 1;
                    false
                }
                'x' | '+' | '-' | '*' | '/' | '=' | '%' | ')' if state == 1 => {
                    token.push(Token::Int(digit));
                    digit = 0;
                    state = 0;
                    true
                }
                '(' if state == 1 => {
                    token.push(Token::Int(digit));
                    digit = 0;
                    token.push(Token::Multiply);
                    state = 0;
                    true
                }
                'x' | '+' | '-' | '*' | '/' | '=' | '%' | ')' if state == 2 => {
                    token.push(Token::Double(digit, frac));
                    digit = 0;
                    frac = 1;
                    state = 0;
                    true
                }
                '(' if state == 2 => {
                    token.push(Token::Double(digit, frac));
                    digit = 0;
                    frac = 1;
                    token.push(Token::Multiply);
                    state = 0;
                    true
                }
                'x' if state == 0 => {
                    token.push(Token::X);
                    false
                }
                '+' if state == 0 => {
                    token.push(Token::Plus);
                    false
                }
                '-' if state == 0 => {
                    token.push(Token::Minus);
                    false
                }
                '*' if state == 0 => {
                    token.push(Token::Multiply);
                    false
                }
                '/' if state == 0 => {
                    token.push(Token::Divide);
                    false
                }
                '%' if state == 0 => {
                    token.push(Token::Percent);
                    false
                }
                '=' if state == 0 => {
                    token.push(Token::Equal);
                    false
                }
                '(' if state == 0 => {
                    token.push(Token::Left);
                    false
                }
                ')' if state == 0 => {
                    token.push(Token::Right);
                    false
                }
                c => {
                    println!("Unknown Token `{c}`({}) at state {state}!!!", c as i64);
                    error = true;
                    false
                }
            } {}
        }
        match state {
            1 => token.push(Token::Int(digit)),
            2 => token.push(Token::Double(digit, frac)),
            _ => (),
        }
        if !token
            .iter()
            .any(|x| (if let &Token::Equal = x { true } else { false }))
        {
            for i in 0..token.len() {
                if let Token::X = token[i] {
                    token[i] = Token::C;
                }
            }
        }
        Parser { token, error }
    }
}

pub fn parse(expression: String) -> String {
    let parser = Parser::new(expression);
    if parser.error {
        return String::new();
    }
    let calc = Caculator::new(parser);
    if calc.error{
        return String::new();
    }
    let res = calc.solve();
    match res {
        Token::Int(x) => x.to_string(),
        Token::Double(x, y) => compute::xEy(x, y),
        Token::Fractor(x, y) => format!("{x}_{y}"),
        Token::X => "x?".to_string(),
        Token::C => ">?".to_string(),
        Token::Plus => "+".to_string(),
        Token::Minus => "-".to_string(),
        Token::Multiply => "*".to_string(),
        Token::Divide => "/".to_string(),
        Token::Percent => "%".to_string(),
        Token::Left => "(".to_string(),
        Token::Right => ")".to_string(),
        Token::Equal => "=".to_string(),
        Token::GreatThan => ">".to_string(),
        Token::LessThan => "<".to_string(),
    }
}

#[derive(Debug, Clone)]
struct Tree {
    op: Token,
    leaf: Vec<Tree>,
    error: bool,
    x_leaf_index: i64,
}

impl Tree {
    /// pirority:
    /// - 0 =><
    /// - 1 +-
    /// - 2 */
    /// - 3 %  
    /// - 4 ()
    /// - 5 num
    fn new(tokens: Vec<Token>) -> (Self, bool) {
        let tokens = &mut tokens.into_iter();
        let mut slf = Self::from_token_iter(tokens);
        if tokens.next().is_some() {
            println!("Tokens not consumed!");
            slf.error = true;
        }
        let code = slf.check();
        if code != 3 && code != 4 {
            slf.error = true;
        }
        match slf.op {
            Token::Equal | Token::C => (),
            _ => slf.error = true,
        }
        (slf, code == 4)
    }
    fn from_token_iter(tokens: &mut std::vec::IntoIter<Token>) -> Self {
        let Some(t) = tokens.next() else {
            return Self::null();
        };
        let mut slf = Self::from_token(t);
        if let Token::Right = slf.op {
            return slf;
        } else if let Token::Left = slf.op {
            slf = Self::from_token_iter(tokens);
        }
        while let Some(token) = tokens.next() {
            if let Token::Right = token {
                return slf;
            } else if let Token::Left = token {
                slf.insert_brackets(Self::from_token_iter(tokens));
            } else if token.pirority() > slf.op.pirority() {
                slf.insert(token);
            } else {
                slf = Self {
                    op: token,
                    leaf: vec![slf],
                    error: false,
                    x_leaf_index: -1,
                }
            }
        }
        slf
    }
    fn from_token(token: Token) -> Self {
        Self {
            op: token,
            leaf: Vec::new(),
            error: false,
            x_leaf_index: -1,
        }
    }
    fn null() -> Self {
        let mut slf = Self::from_token(Token::NULL);
        slf.error = true;
        slf
    }
    fn insert_brackets(&mut self, t: Tree) {
        if self.leaf.len() < 2 {
            self.leaf.push(t);
        } else {
            self.leaf.last_mut().unwrap().insert_brackets(t);
        }
    }
    fn insert(&mut self, token: Token) {
        if self.leaf.len() < 2 {
            self.leaf.push(Self::from_token(token));
        } else if token.pirority() > self.leaf.last().unwrap().op.pirority() {
            self.leaf.last_mut().unwrap().insert(token);
        } else {
            let last = self.leaf.pop().unwrap();
            self.leaf.push(Self {
                op: token,
                leaf: vec![last],
                error: false,
                x_leaf_index: -1,
            });
        }
    }
    fn is_error(&self) -> bool {
        if self.error {
            return true;
        }
        for i in &self.leaf {
            if i.is_error() {
                return true;
            }
        }
        false
    }

    // return 1 when there is one =
    // return 2 when there is one x(f)
    // return 3 when there is one x(t)
    // return 4 when there is one x(f) and =
    // return -1 when there is error on number of = or x
    fn check(&mut self) -> i64 {
        let mut code = match self.op {
            Token::GreatThan | Token::LessThan | Token::Right => -1,
            Token::C => 3,
            Token::X => {
                if self.leaf.is_empty() {
                    return 2;
                }
                self.error = true;
                2
            }
            Token::Equal => {
                if self.leaf.len() == 1 {
                    if self.leaf[0].check() == 0 {
                        // x is on right
                        self.leaf.push(Tree::from_token(Token::X));
                        self.x_leaf_index = 1;
                        return 4;
                    } else {
                        self.error = true;
                        return -1;
                    }
                } else {
                    1
                }
            }
            Token::Int(_) | Token::Fractor(_, _) | Token::Double(_, _) => {
                if self.leaf.is_empty() {
                    return 0;
                }
                self.error = true;
                0
            }
            _ => 0,
        };
        for (index, t) in self.leaf.iter_mut().enumerate() {
            let c = t.check();
            if c == 2 || c == 4 {
                self.x_leaf_index = index as i64;
            }
            if code == 1 && c == 2 || code == 2 && c == 1 {
                code = 4;
            } else if code != 0 && c != 0 {
                self.error = true;
                code = -1;
            } else {
                code += c;
            }
        }
        code
    }

    fn solve_equation(&self) -> (i64, i64) {
        if self.leaf.len() != 2 {
            return (0, 1);
        }
        let Token::Equal = self.op else {
            return (0, 1);
        };
        match self.x_leaf_index {
            0 => self.leaf[0].solve_equal(self.leaf[1].calc()),
            1 => self.leaf[1].solve_equal(self.leaf[0].calc()),
            _ => (0, 1),
        }
    }
    fn solve_equal(&self, res: (i64, i64)) -> (i64, i64) {
        if let Token::X = self.op {
            return (res.0, res.1);
        }
        if self.leaf.len() != 2 {
            return (0, 1);
        }
        let (leaf, val) = match self.x_leaf_index {
            0 => (&self.leaf[0], self.leaf[1].calc()),
            1 => (&self.leaf[1], self.leaf[0].calc()),
            _ => return (0, 1),
        };
        match self.op {
            // val + x = res
            Token::Plus => leaf.solve_equal(compute::minus(res, val)),
            // x - val = res
            Token::Minus if self.x_leaf_index == 0 => leaf.solve_equal(compute::plus(res, val)),
            // val - x = res
            Token::Minus if self.x_leaf_index == 1 => leaf.solve_equal(compute::minus(val, res)),
            // val * x = res
            Token::Multiply => leaf.solve_equal(compute::divide(res, val)),
            // x / val = res
            Token::Divide if self.x_leaf_index == 0 => leaf.solve_equal(compute::times(res, val)),
            // val / x = res
            Token::Divide if self.x_leaf_index == 1 => leaf.solve_equal(compute::divide(val, res)),
            _ => (0, 1),
        }
    }

    fn solve_inequality(&self) -> Token {
        if self.leaf.len() != 2 {
            return Token::NULL;
        }
        let Token::C = self.op else {
            return Token::NULL;
        };
        let left = self.leaf[0].calc();
        let right = self.leaf[1].calc();
        Token::cmp(left, right)
    }
    fn calc(&self) -> (i64, i64) {
        if self.leaf.is_empty() {
            return self.op.get_num();
        }
        let (l, r) = self.getPair();
        match self.op {
            Token::Plus => compute::plus(l, r),
            Token::Minus => compute::minus(l, r),
            Token::Multiply => compute::times(l, r),
            Token::Divide => compute::divide(l, r),
            _ => (0, 1),
        }
    }
    fn getPair(&self) -> (compute::Num, compute::Num) {
        let left = self
            .leaf
            .get(0)
            .and_then(|t| Some(t.calc()))
            .unwrap_or_default();
        let right = self
            .leaf
            .get(1)
            .and_then(|t| Some(t.calc()))
            .unwrap_or_default();
        // let p = left.get_num_pirority().max(right.get_num_pirority());
        // // let left = left.get_num_at_p(p);
        // // let right = right.get_num_at_p(p);
        // let left = left.get_num();
        // let right = right.get_num();
        (left, right)
    }
}

#[derive(Debug, Clone)]
struct Caculator {
    parser: Parser,
    tree: Tree,
    is_equation: bool,
    error: bool,
}

impl Caculator {
    fn new(parser: Parser) -> Self {
        let (tree, is_equation) = Tree::new(parser.token.clone());
        let error = tree.is_error();
        Self {
            parser,
            error,
            tree,
            is_equation,
        }
    }
    fn solve(&self) -> Token {
        if self.error {
            Token::X
        } else if self.is_equation {
            Token::from_num(self.tree.solve_equation())
        } else {
            self.tree.solve_inequality()
        }
    }
}

#[test]
fn test_parser() {
    let tic = std::time::SystemTime::now();
    let parser = Parser::new("1+2-3.4*4+(7/3*(x-2-6)-6*7)+8=9*10".to_string());
    // let parser = Parser::new("(5*x)=4".to_string());
    // let parser = Parser::new("x=4*5".to_string());
    // let parser = Parser::new("5/(x-(((3-6))))*4=9".to_string());
    // let parser = Parser::new("2*100-40*0.9991x10*(20-4)".to_string());
    // let parser = Parser::new("1x2/(1+1)".to_string());
    assert!(parser.error == false);
    let calc = Caculator::new(parser);
    // println!("{:#?}", calc);
    assert!(calc.error == false);
    let res = calc.solve();
    println!("{:#?}", res);
    println!("Time costs: {:?}", tic.elapsed().unwrap());
}
