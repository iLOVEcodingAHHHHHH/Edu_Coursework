from typing import Optional
from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel, Field
from starlette import status

app = FastAPI()

class Book:
    id: int
    title: str
    author: str
    description: str
    rating: int

    def __init__(self, id, published_date, title, author, description, rating):
        self.id = id
        self.published_date = published_date
        self.title = title
        self.author = author
        self.description = description
        self.rating = rating


class BookRequest(BaseModel):
    id: Optional[int] = Field(description='ID is not needed on create', default=None)
    published_date: int = Field(min_length=4, max_length=4)
    title: str =Field(min_length=3)
    author: str =Field(min_length=1)
    description: str = Field(min_length=1, max_length=100)
    rating: int = Field(gt=0, lt=6)

    model_config = {
        "json_schema_extra": {
            "example": {
                "published_date": 1990,
                "title": "A new book",
                "author": "codingwithnick",
                "description": "A new description",
                "rating": 5
            }
        }
    }

BOOKS = [
    Book(1, 2000, 'Computer Science Pro', 'codingwithroby', 'A very nice book!', 5),
    Book(2, 2001, 'Be Fast with FastAPI', 'codingwithroby', 'A great book!', 5),
    Book(3, 2002, 'Master Endpoints', 'codingwithroby', 'An awesome book!', 5),
    Book(4, 2000, 'HP1', 'Author 1', 'Book Description', 2),
    Book(5, 2001, 'HP2', 'Author 2', 'Book Description', 3),
    Book(6, 2002, 'HP3', 'Author 3', 'Book Description', 1),
]


@app.get("/books", status_code=status.HTTP_200_OK)
async def read_all_books():
    return BOOKS


@app.get("/books/publish/")
async def get_book_by_date(published_year: int = Query(min_length=4, max_length=4)):
    returned_books = []
    for book in BOOKS:
        if book.published_date == published_year:
            returned_books.append(book)
    return returned_books

@app.get("/books/{book_id}", status_code=status.HTTP_200_OK)
async def get_book_by_id(book_id: int = Path(gt=0)):
    for book in BOOKS:
        if book.id == book_id:
            return book
    raise HTTPException(404, 'Item not found')

@app.get("/books/", status_code=status.HTTP_200_OK)
async def read_book_by_rating(book_rating: int = Query(gt=0, lt=6)):
    books_to_return = []
    for book in BOOKS:
        if book.rating == book_rating:
            books_to_return.append(book)
    return books_to_return


@app.post("/create-book", status_code=status.HTTP_201_CREATED)
async def create_book(book_request: BookRequest):
    new_book = Book(**book_request.model_dump())
    BOOKS.append(find_book_id(new_book))

def find_book_id(book: Book):
    book.id = 1 if len(BOOKS) == 0 else BOOKS[-1].id + 1
    return book

@app.put("/books/update_book", status_code=status.HTTP_204_NO_CONTENT)
async def update_book(book: BookRequest):
    book_changed = False
    for i in range(len(BOOKS)):
        if BOOKS[i].id == book.id:
            BOOKS[i] = book
            book_changed = True
    if not book_changed:
        raise HTTPException(404, 'Item not found')


@app.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(book_id: int = Path(gt=0)):
    book_changed = False
    for i in range(len(BOOKS)):
        if BOOKS[i].id == book_id:
            BOOKS.pop(i)
            break
    if not book_changed:
        raise HTTPException(404, 'Item not found')