const mysql = require('mysql2');

const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'Root@1234',
    database: 'rag_app'
});

db.connect(err => {
    if (err) {
        console.error('DB connection failed:', err);
    } else {
        console.log('MySQL Connected ✅');
    }
});

module.exports = db;